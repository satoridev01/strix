"""Dependency-free smoke checks for the Satori runtime adapter."""

from __future__ import annotations

import asyncio
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class LocalBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        backends = Path(__file__).resolve().parents[1] / "strix/runtime/backends.py"
        self.module = runpy.run_path(str(backends))

    def test_local_session_starts_without_docker(self) -> None:
        created: dict[str, object] = {}

        class Session:
            async def start(self) -> None:
                created["started"] = True

        class Client:
            async def create(self, *, options: object, manifest: object) -> Session:
                created.update(options=options, manifest=manifest)
                return Session()

        class Options:
            def __init__(self, *, exposed_ports: tuple[int, ...]) -> None:
                self.exposed_ports = exposed_ports

        unix_local = types.ModuleType("agents.sandbox.sandboxes.unix_local")
        unix_local.UnixLocalSandboxClient = Client  # type: ignore[attr-defined]
        unix_local.UnixLocalSandboxClientOptions = Options  # type: ignore[attr-defined]
        modules = {
            name: types.ModuleType(name)
            for name in ("agents", "agents.sandbox", "agents.sandbox.sandboxes")
        }
        modules[unix_local.__name__] = unix_local
        manifest = object()

        with patch.dict(sys.modules, modules):
            client, session = asyncio.run(
                self.module["get_backend"]("local")(
                    image="unused", manifest=manifest, exposed_ports=()
                )
            )

        self.assertIsInstance(client, Client)
        self.assertIsInstance(session, Session)
        self.assertIs(created["manifest"], manifest)
        self.assertEqual(created["options"].exposed_ports, ())
        self.assertTrue(created["started"])
        self.assertFalse(self.module["backend_supports_bind_mounts"]("local"))

    def test_local_backend_rejects_bind_mounts(self) -> None:
        with self.assertRaisesRegex(ValueError, "bind mounts"):
            asyncio.run(
                self.module["get_backend"]("local")(
                    image="unused",
                    manifest=object(),
                    exposed_ports=(),
                    bind_mounts=[{"source": "/private/workspace", "target": "/workspace"}],
                )
            )


if __name__ == "__main__":
    unittest.main()
