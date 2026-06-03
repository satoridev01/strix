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

    # Optional audit trail: log every command the agent executes through the
    # sandbox (proves tools actually run locally). Enable with STRIX_EXEC_LOG=path.
    log_path = os.environ.get("STRIX_EXEC_LOG")
    if log_path:
        _orig_exec = session.exec

        async def _logged_exec(*command, **kwargs):
            try:
                with open(log_path, "a") as fh:
                    fh.write(" ".join(str(c) for c in command) + "\n")
            except Exception:
                pass
            return await _orig_exec(*command, **kwargs)

        session.exec = _logged_exec

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
    argv = ["strix", "-n", "--target", host]
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
        return e.code
    except Exception:
        print("[wrapper] strix raised an exception:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
