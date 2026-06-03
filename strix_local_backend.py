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
    return client, session


def _main_module():
    # strix.interface.__init__ re-exports the `main` function, shadowing the
    # `strix.interface.main` submodule on attribute access. Pull the real module
    # out of sys.modules via importlib.
    return importlib.import_module("strix.interface.main")


def install():
    # 1) register the local backend
    from strix.runtime.backends import register_backend
    register_backend("local", local_backend)

    # 2) neutralize the Docker preflight in the CLI entrypoint
    M = _main_module()
    M.check_docker_installed = lambda *a, **k: None
    M.pull_docker_image = lambda *a, **k: None

    # 3) stub the in-container Caido sidecar bootstrap
    import strix.runtime.session_manager as SM

    async def _no_caido(*a, **k):
        return None

    SM.bootstrap_caido = _no_caido

    os.environ["STRIX_RUNTIME_BACKEND"] = "local"


def main():
    install()
    M = _main_module()
    # Target + scan mode come from env (set by the Satori playbook); any extra
    # raw strix flags can be passed via STRIX_EXTRA_ARGS (space-separated).
    host = os.environ.get("STRIX_TARGET", "")
    scan_mode = os.environ.get("STRIX_SCAN_MODE", "quick")
    extra = os.environ.get("STRIX_EXTRA_ARGS", "").split()
    sys.argv = ["strix", "-n", "--scan-mode", scan_mode, "--target", host, *extra]
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
