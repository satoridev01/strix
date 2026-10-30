"""Extra files must land in the session's real workspace, whatever its root."""

from __future__ import annotations

from pathlib import Path

from agents.sandbox.manifest import Manifest

from strix.runtime.backends import get_backend
from strix.runtime.session_manager import build_extra_file_archive, stage_extra_files


async def test_the_local_backend_places_extra_files_in_its_own_workspace() -> None:
    archive = build_extra_file_archive(
        [{"workspace_path": "/workspace/prior-findings.json", "content": b"[]\n"}]
    )
    assert archive is not None
    client, session = await get_backend("local")(
        image="unused", manifest=Manifest(), exposed_ports=()
    )
    try:
        root = Path(str(session.state.manifest.root))
        assert root != Path("/workspace")

        await stage_extra_files(session, archive)

        assert (root / "prior-findings.json").read_bytes() == b"[]\n"
        assert not (root / ".strix-extra-files.tar").exists()
    finally:
        await client.delete(session)
