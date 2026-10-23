"""Strix 'local' runtime backend — runs the pentest agent's tools directly
inside the CURRENT container (the Satori sandbox) instead of spawning a sibling
Docker sandbox. Built on the openai-agents SDK's own `unix_local` sandbox, so
all SDK-internal session machinery (materialization, concurrency limits, etc.)
is satisfied by the real implementation.

It registers STRIX_RUNTIME_BACKEND=local, neutralizes the Docker preflight, and
stubs the in-container Caido proxy sidecar (which doesn't exist locally).
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import time
import traceback

_PROXY_KEYS = (
    "http_proxy", "https_proxy", "all_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
)


async def local_backend(*, image, manifest, exposed_ports):
    """Mirror of Strix's _docker_backend, but on the SDK's unix_local sandbox."""
    from agents.sandbox.manifest import Environment
    from agents.sandbox.sandboxes.unix_local import (
        UnixLocalSandboxClient,
        UnixLocalSandboxClientOptions,
    )

    # Strip the Caido proxy env Strix injects — the sidecar isn't running here.
    if manifest is not None:
        try:
            env = dict(manifest.environment.value)
            changed = False
            for k in _PROXY_KEYS:
                if k in env:
                    env.pop(k, None)
                    changed = True
            if changed:
                manifest = manifest.model_copy(
                    update={"environment": Environment(value=env)}
                )
        except Exception:
            pass

    client = UnixLocalSandboxClient()
    options = UnixLocalSandboxClientOptions(exposed_ports=tuple(exposed_ports))
    session = await client.create(options=options, manifest=manifest)
    await session.start()
    return client, session


def _main_module():
    # strix.interface.__init__ re-exports the `main` function, shadowing the
    # `strix.interface.main` submodule on attribute access. Pull the real module
    # out of sys.modules via importlib.
    return importlib.import_module("strix.interface.main")


class IncompatibleStrixError(RuntimeError):
    """Raised when this shim's assumptions no longer hold against the installed
    strix-agent / openai-agents. Fail LOUD instead of silently no-op'ing a
    monkeypatch (which would otherwise resurface as a confusing Docker error)."""


def _versions():
    import importlib.metadata as md

    def v(name):
        try:
            return md.version(name)
        except Exception:
            return "?"

    return v("strix-agent"), v("openai-agents")


def _require(cond, msg):
    if not cond:
        sv, av = _versions()
        raise IncompatibleStrixError(
            f"{msg} (strix-agent={sv}, openai-agents={av}). "
            "The local-backend shim is pinned to these internals; update "
            "satoridev01/strix to match the installed versions."
        )


def _quiet_litellm():
    # Strix retries failed model calls; on each failure LiteLLM dumps a noisy
    # "Provider List: https://docs.litellm.ai/..." banner to stdout. Silence it.
    try:
        import litellm

        litellm.suppress_debug_info = True
        litellm.set_verbose = False
    except Exception:
        pass
    for name in ("LiteLLM", "litellm"):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass


def retry_midstream_cuts():
    """Make a model call that dies mid-response retryable.

    The SDK retries a streamed model call only while nothing but `response.created` / `in_progress` has
    reached the agent loop: once text is out, a failure is final and ends the whole pentest. A provider
    that drops mid-response is reported by OpenRouter in-band, after the first token, where it cannot
    switch providers; one such cut threw away a 41-minute scan. So each response is held back until it is
    complete: a cut then happens before anything was emitted, Strix's own retry settings ask again, and
    OpenRouter routes the new request to a provider that is up. The loop reads the finished response,
    never the deltas. Disable with STRIX_ATOMIC_STREAM=0.
    """
    if os.environ.get("STRIX_ATOMIC_STREAM", "1").lower() in ("0", "false", "no", "off", ""):
        return
    import functools

    try:
        from agents.extensions.models.litellm_model import LitellmModel
    except Exception as e:
        print(f"[shim] model streams not held until complete: {e!r}", flush=True)
        return
    _require(hasattr(LitellmModel, "stream_response"), "LitellmModel.stream_response missing")
    stream_response = LitellmModel.stream_response
    if getattr(stream_response, "_strix_atomic", False):
        return

    @functools.wraps(stream_response)
    async def whole_response(self, *args, **kwargs):
        try:
            events = [event async for event in stream_response(self, *args, **kwargs)]
        except Exception as e:
            print(
                f"[shim] model call cut before it finished ({type(e).__name__}); "
                "the retry settings decide whether it is asked again",
                flush=True,
            )
            raise
        for event in events:
            yield event

    whole_response._strix_atomic = True
    LitellmModel.stream_response = whole_response


# Root-agent turns from which it is told they are running out. Healthy roots end by turn 440 (p99 395);
# every root that reached the cap of 500 had never called finish_scan.
_TURN_WARNINGS = (450, 475)

_CLOSE = (
    "say in the methodology what was not covered, "
    "and do not mention turns, budget, agents or tooling in the report"
)
# Short and imperative on purpose: a longer, generic note had the production model go on exploring.
_NOTES = {
    ("first", False): (
        "Turn budget: {left} turns left, then this scan is cut off and nothing you have not reported is kept. "
        "No sub-agent is running. Stop exploring and call finish_scan now with your report: " + _CLOSE + "."
    ),
    ("first", True): (
        "Turn budget: {left} turns left, then this scan is cut off and nothing you have not reported is kept. "
        "Still running: {running}. Stop exploring. Send each one a message to file its findings and finish "
        "(send_message_to_agent), wait for them (wait_for_message), then call finish_scan with your report: "
        + _CLOSE + "."
    ),
    ("final", False): (
        "FINAL WARNING: {left} turns left. No sub-agent is running. "
        "Your next call must be finish_scan; " + _CLOSE + "."
    ),
    ("final", True): (
        "FINAL WARNING: {left} turns left. Still running: {running}. "
        "Stop each one with stop_agent now, then call finish_scan; " + _CLOSE + "."
    ),
}

# RiskPoint stops an ACTIVE run at 6 h and keeps nothing from a run stopped that way, so the root is told to
# close an hour before.
_CLOCK_WARNING_SECONDS = 5 * 60 * 60
_CLOCK_NOTES = {
    False: "Time budget: this scan is stopped within the hour and nothing you have not reported is kept. "
    "No sub-agent is running. Stop exploring and call finish_scan now with your report: " + _CLOSE + ".",
    True: "Time budget: this scan is stopped within the hour and nothing you have not reported is kept. "
    "Still running: {running}. Stop each one with stop_agent now, then call finish_scan; " + _CLOSE + ".",
}

_turn_warnings = {"sent": False}


def _running_sub_agents(context):
    """The sub-agents finish_scan would refuse to leave behind, as 'name (id)'; None when the graph is unreadable."""
    try:
        coordinator, root = context["coordinator"], context["agent_id"]
        return [
            f"{str(coordinator.names.get(a, a))[:40]} ({a})".replace("\n", " ")
            for a, status in coordinator.statuses.items()
            if a != root and status in ("running", "waiting")
        ]
    except Exception:
        return None


def warn_root_before_turn_cap():
    """Tell the root agent its turns are running out, so it closes the scan instead of dying with it.

    Only the root's MaxTurnsExceeded ends the pentest, and in every run that hit it the root had not called
    finish_scan. From its turn 450, and again from 475, each model call the root makes carries a note with the
    turns left and the one thing to do, which depends on whether sub-agents are still running; it rides on that
    call only and never enters the agent's history. After 5 h the root gets the same kind of note, to close
    before RiskPoint stops the run at 6 h. Turns are counted over the whole run of the root, so a
    forced continuation does not silence the note. A scan that was warned and then ended with no finding is
    reported as incomplete (incomplete_after_warning), not clean. Disable with STRIX_TURN_WARNINGS=0.
    """
    if os.environ.get("STRIX_TURN_WARNINGS", "1").lower() in ("0", "false", "no", "off", ""):
        return
    import functools

    try:
        from agents.run_config import ModelInputData, RunConfig
        from strix.core.inputs import DEFAULT_MAX_TURNS
    except Exception as e:
        print(f"[shim] root agent not warned before its turn cap: {e!r}", flush=True)
        return
    first, last = _TURN_WARNINGS
    _require(
        "call_model_input_filter" in RunConfig.__dataclass_fields__,
        "RunConfig.call_model_input_filter missing",
    )
    _require(DEFAULT_MAX_TURNS > last, f"turn cap {DEFAULT_MAX_TURNS} is not above the warnings {_TURN_WARNINGS}")
    if getattr(RunConfig.__init__, "_strix_turn_warnings", False):
        return

    turns = {}  # model calls the root has made so far
    started = time.monotonic()

    def add_note(data):
        context = data.context if isinstance(data.context, dict) else {}
        if context.get("parent_id") is not None:
            return data.model_data
        agent_id = context.get("agent_id")
        turns[agent_id] = turn = turns.get(agent_id, 0) + 1
        late = time.monotonic() - started >= _CLOCK_WARNING_SECONDS
        if turn < first and not late:
            return data.model_data
        if not _turn_warnings["sent"] and late and turn < first:
            print(f"[shim] root agent warned at {_CLOCK_WARNING_SECONDS // 3600} h", flush=True)
        _turn_warnings["sent"] = True
        if turn in (first, last):
            print(f"[shim] root agent warned at turn {turn} of {DEFAULT_MAX_TURNS}", flush=True)
        running = _running_sub_agents(context)
        names = ", ".join(running) if running else "the sub-agents you started (see view_agent_graph)"
        if turn < first:
            note = _CLOCK_NOTES[running != []].format(running=names)
        else:
            note = _NOTES["final" if turn >= last else "first", running != []].format(
                left=max(DEFAULT_MAX_TURNS - turn, 0), running=names
            )
        return ModelInputData(
            input=[*data.model_data.input, {"role": "user", "content": note}],
            instructions=data.model_data.instructions,
        )

    init = RunConfig.__init__

    @functools.wraps(init)
    def with_warnings(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if self.call_model_input_filter is None:
            self.call_model_input_filter = add_note

    with_warnings._strix_turn_warnings = True
    RunConfig.__init__ = with_warnings


def incomplete_after_warning(code):
    """A root that was warned it was out of turns and then ended clean did not cover the target: report it as
    incomplete (3), which the playbook fails and RiskPoint keeps only if findings were filed. Findings (2) and
    crashes (1) keep their code."""
    if _turn_warnings["sent"] and not code:
        print(
            "[shim] the root agent ran short of turns and filed no finding: "
            "reporting the run as incomplete (exit 3)",
            flush=True,
        )
        return 3
    return code


def install():
    sv, av = _versions()
    print(f"[shim] strix-agent={sv} openai-agents={av} -> backend=local", flush=True)

    _quiet_litellm()

    # 0) the SDK must ship the unix_local (subprocess) sandbox we build on
    try:
        import agents.sandbox.sandboxes.unix_local  # noqa: F401
    except Exception as e:  # pragma: no cover
        _require(False, f"openai-agents has no unix_local sandbox ({e!r})")

    # 1) register the local backend
    from strix.runtime.backends import register_backend
    register_backend("local", local_backend)

    # 2) neutralize the Docker preflight in the CLI entrypoint — assert the
    #    patch targets exist so an upstream rename fails here, not mid-scan.
    M = _main_module()
    _require(hasattr(M, "check_docker_installed"), "main.check_docker_installed missing")
    _require(hasattr(M, "pull_docker_image"), "main.pull_docker_image missing")
    M.check_docker_installed = lambda *a, **k: None
    M.pull_docker_image = lambda *a, **k: None

    # 3) stub the in-container Caido sidecar bootstrap
    import strix.runtime.session_manager as SM
    _require(hasattr(SM, "bootstrap_caido"), "session_manager.bootstrap_caido missing")

    async def _no_caido(*a, **k):
        return None

    SM.bootstrap_caido = _no_caido

    # 4) optional audit trail — log every command the agent runs through the
    #    sandbox. _exec_internal is the chokepoint all tool execution funnels
    #    through, so patch it at the class level. Enable with STRIX_EXEC_LOG=path.
    log_path = os.environ.get("STRIX_EXEC_LOG")
    if log_path:
        from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxSession

        if not getattr(UnixLocalSandboxSession, "_strix_exec_logged", False):
            _orig_exec_internal = UnixLocalSandboxSession._exec_internal

            async def _logged_exec_internal(self, *command, **kwargs):
                try:
                    with open(log_path, "a") as fh:
                        fh.write(" ".join(str(c) for c in command) + "\n")
                except Exception:
                    pass
                return await _orig_exec_internal(self, *command, **kwargs)

            UnixLocalSandboxSession._exec_internal = _logged_exec_internal
            UnixLocalSandboxSession._strix_exec_logged = True

    # 5) Anthropic prompt caching. Strix never sets cache_control, so the large
    #    static system prompt (~55K tokens) is billed at full input price on
    #    EVERY turn — Claude runs cost ~15-20x more than they should. Anthropic
    #    (incl. via OpenRouter) only caches when cache_control breakpoints are
    #    present, so we use litellm's cache_control_injection_points to mark the
    #    system message. Cache reads are 0.1x input price. Claude-only; other
    #    providers (Gemini/OpenAI) cache automatically and are left untouched.
    #    Disable with STRIX_ANTHROPIC_CACHE=0.
    if os.environ.get("STRIX_ANTHROPIC_CACHE", "1").lower() not in ("0", "false", "no", "off", ""):
        try:
            import functools
            import litellm

            # Two breakpoints: the static system prompt (~55K) AND the last
            # message, so the growing conversation prefix also caches turn-to-turn.
            _injection = [
                {"location": "message", "role": "system"},
                {"location": "message", "index": -1},
            ]

            def _with_cache(fn):
                @functools.wraps(fn)
                def wrapper(*a, **kw):
                    model = str(kw.get("model", "")).lower()
                    if ("anthropic" in model or "claude" in model) and \
                            "cache_control_injection_points" not in kw:
                        kw["cache_control_injection_points"] = _injection
                    return fn(*a, **kw)
                return wrapper

            for _name in ("acompletion", "completion"):
                _orig = getattr(litellm, _name, None)
                if _orig is not None and not getattr(_orig, "_strix_cache_wrapped", False):
                    _wrapped = _with_cache(_orig)
                    _wrapped._strix_cache_wrapped = True
                    setattr(litellm, _name, _wrapped)
        except Exception:
            pass

    # 6) A model call cut mid-response is retried (see retry_midstream_cuts).
    retry_midstream_cuts()

    # 7) The root agent is warned before its turns run out (see warn_root_before_turn_cap).
    warn_root_before_turn_cap()

    os.environ["STRIX_RUNTIME_BACKEND"] = "local"


def main():
    install()
    M = _main_module()
    # Target + scan mode come from env (set by the Satori playbook); any extra
    # raw strix flags can be passed via STRIX_EXTRA_ARGS (space-separated).
    host = os.environ.get("STRIX_TARGET", "").strip()
    if not host:
        print("[shim] ERROR: STRIX_TARGET is empty — pass -d HOST=...", flush=True)
        return 1
    # STRIX_TARGET may contain newline-separated domains (multi-target scans); each
    # becomes its own --target flag so Strix evaluates them in parallel.
    targets = [t.strip() for t in host.splitlines() if t.strip()]
    argv = ["strix", "-n"]
    for t in targets:
        argv += ["--target", t]
    # Empty/unset scan mode → omit the flag so strix uses its own default (deep),
    # rather than passing --scan-mode "" (which would error).
    scan_mode = os.environ.get("STRIX_SCAN_MODE", "").strip()
    if scan_mode:
        argv += ["--scan-mode", scan_mode]
    argv += os.environ.get("STRIX_EXTRA_ARGS", "").split()
    sys.argv = argv
    try:
        M.main()
    except SystemExit as e:
        print(f"[wrapper] strix exited with code: {e.code}")
        return incomplete_after_warning(e.code)
    except Exception:
        print("[wrapper] strix raised an exception:")
        traceback.print_exc()
        return 1
    return incomplete_after_warning(0)


if __name__ == "__main__":
    sys.exit(main() or 0)
