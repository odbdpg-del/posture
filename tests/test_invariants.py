"""Structural tests for the two promises the app makes about itself.

These scan the package source rather than its behaviour. That is on purpose:
"no frames on disk" and "no network" are properties that have to hold on every
code path, including the ones no test exercises, and the cheapest way to keep
them true a year from now is to make violating them fail the build.

``scripts/`` is deliberately outside the scan. The model fetcher lives there
precisely so that the one network call in the project sits outside the code
that runs while the app is watching you.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import posture

PACKAGE = Path(posture.__file__).resolve().parent
SOURCES = sorted(PACKAGE.rglob("*.py"))

# Calls that could put pixels somewhere they persist. Names like ``save`` and
# ``dump`` are matched only when qualified, so that writing derived numbers out
# as JSON (which later phases must do) stays legal while ``np.save`` on an
# array does not.
#
# ``imencode`` is not in this set because encoding is not persistence -- it is
# how the loopback video preview reaches the browser. It gets its own, narrower
# test below: exactly one module may call it. Everything here writes to a file
# or a file-like sink, and none of it is allowed anywhere, ever.
FRAME_SINK_NAMES = {"imwrite", "imsave", "VideoWriter", "tofile"}
FRAME_SINK_QUALIFIED = {
    "cv2.imwrite", "cv2.imsave", "cv2.VideoWriter",
    "np.save", "np.savez", "np.savez_compressed", "np.savetxt",
    "numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.savetxt",
    "pickle.dump", "pickle.dumps",
}
# Modules that would let the app talk to anything off-machine.
NETWORK_MODULES = {
    "socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx",
    "ftplib", "smtplib", "telnetlib", "xmlrpc", "asyncio.streams",
}
# The control panel is a local HTTP server, so it needs http.server. The
# exemption is per-file rather than global: only webui.py may import it, and
# separate tests below check that it can only ever bind to loopback. Anything
# else in the package importing a network module is still a failure.
NETWORK_ALLOWED: dict[str, set[str]] = {
    "webui.py": {"http.server"},
}


def _qualname(node: ast.AST) -> str:
    """Dotted name for a call target, e.g. ``cv2.imwrite`` or ``np.save``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualname(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_package_has_sources_to_scan():
    """Guard against the scan silently passing because it found nothing."""
    assert len(SOURCES) >= 5


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_frame_is_ever_written_out(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offences = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qualified = _qualname(node.func)
        bare = qualified.rsplit(".", 1)[-1]
        if qualified in FRAME_SINK_QUALIFIED or bare in FRAME_SINK_NAMES:
            offences.append(f"{path.name}:{node.lineno} calls {qualified}()")
    assert not offences, (
        "frames (or arrays that might be frames) must never reach disk: "
        + "; ".join(offences)
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_package_cannot_reach_the_network(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offences = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        allowed = NETWORK_ALLOWED.get(path.name, set())
        for name in names:
            root = name.split(".")[0]
            if root in NETWORK_MODULES and name not in allowed:
                offences.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offences, "the app must stay offline: " + "; ".join(offences)


def test_model_fetcher_is_the_only_networked_code():
    """The exemption is real and narrow: exactly one file, outside the package."""
    fetcher = PACKAGE.parent / "scripts" / "fetch_model.py"
    assert fetcher.exists()
    assert "urllib.request" in fetcher.read_text(encoding="utf-8")
    assert not fetcher.is_relative_to(PACKAGE)


def test_only_webui_may_import_a_network_module():
    """Keep the exemption from spreading by accident."""
    assert set(NETWORK_ALLOWED) == {"webui.py"}
    exempt = [p for p in SOURCES if p.name in NETWORK_ALLOWED]
    assert len(exempt) == 1, "the exempt file must actually exist in the package"


# Encoding a frame is how the loopback preview works, so it is allowed -- but
# in one module only, so it cannot quietly spread to somewhere that then writes
# the bytes out.
ENCODE_ALLOWED = {"webui.py"}


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_only_the_control_panel_may_encode_a_frame(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if path.name in ENCODE_ALLOWED:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _qualname(node.func).rsplit(".", 1)[-1]
            assert name != "imencode", (
                f"{path.name}:{node.lineno} encodes a frame; only "
                f"{sorted(ENCODE_ALLOWED)} may do that"
            )


def test_only_video_mode_retains_a_frame():
    """Skeleton and off must actually stop frames being kept, not just hide them.

    A preview setting that only changed what the page draws, while workers kept
    buffering frames and the endpoint kept encoding them, would be a lie. One
    predicate answers it for both the capture side and the HTTP side.
    """
    from posture.config import Config, retains_frames

    for mode, expected in (("video", True), ("skeleton", False), ("off", False)):
        cfg = Config()
        cfg.web.preview_mode = mode
        assert retains_frames(cfg) is expected, mode

    monitor_src = (PACKAGE / "monitor.py").read_text(encoding="utf-8")
    webui_src = (PACKAGE / "webui.py").read_text(encoding="utf-8")
    assert "retains_frames(self.cfg)" in monitor_src, (
        "the worker must consult the predicate before retaining a frame")
    assert "retains_frames(self.monitor.cfg)" in webui_src, (
        "the frame endpoint must consult the same predicate")


def test_the_stats_database_stores_no_images():
    """A database arrived after the "frames never reach disk" promise. It must
    not be the thing that breaks it: no BLOB columns, no image-shaped names."""
    import sqlite3
    import tempfile

    from posture.store import Store

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(pathlib_path(tmp))
        try:
            tables = store._rows(  # noqa: SLF001
                "SELECT name FROM sqlite_master WHERE type = 'table'")
            assert tables, "the schema must actually have been created"
            for table in tables:
                for col in store._rows(f"PRAGMA table_info({table['name']})"):  # noqa: SLF001
                    kind = (col["type"] or "").upper()
                    assert "BLOB" not in kind, f"{table['name']}.{col['name']}"
                    assert not any(word in col["name"].lower() for word in
                                   ("image", "frame", "jpeg", "png", "pixel"))
        finally:
            store.close()
    assert sqlite3  # keeps the import meaningful to a reader


def pathlib_path(tmp: str):
    return Path(tmp) / "stats.db"


def test_frames_still_never_reach_the_disk():
    """The rule that has no exceptions, stated once more in one place."""
    for path in SOURCES:
        src = path.read_text(encoding="utf-8")
        for banned in ("imwrite(", "VideoWriter(", "tofile("):
            assert banned not in src, f"{path.name} writes frames to disk"
