"""The local control panel: settings, camera assignment and live status.

Networking exemption
--------------------
This is the only module in the package that imports anything network-shaped,
and ``tests/test_invariants.py`` lists it by name for that reason. Two rules
keep the exemption narrow, and both are tested:

* The server binds to loopback only. :func:`serve` raises rather than binding
  anywhere else, so a stray config edit cannot put this on the network.
* No frame is ever written to disk, in any mode. That rule has no exceptions
  and ``tests/test_invariants.py`` still enforces it across the whole package.

The video preview is the one place pixels leave the process. ``/api/frame``
encodes the newest frame and sends it to your own browser over loopback, where
the panel draws the posture geometry over it as SVG. It exists because a stick figure turns out not to
answer the question people actually ask of a camera setup screen -- "is it
pointed at me and can it see me" -- and because aiming a camera you cannot see
through is guesswork. It is a visible setting (``web.video_preview``), and with
it set to anything but "video" -- skeleton or off -- the workers do not retain
frames at all and this endpoint returns nothing, so the reduction is real
rather than cosmetic. ``cv2.imencode`` is permitted in this module and nowhere
else, which the invariant tests check by name.

Everything served is generated locally; there are no external fonts, scripts or
stylesheets, so the page works with the machine offline.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from . import metrics as met
from .config import PREVIEW_MODES, Config, WebConfig, retains_frames
from .devices import check_available, list_devices
from .monitor import Monitor

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"

LOOPBACK = ("127.0.0.1", "::1", "localhost")

# Keys the panel adds to /api/config for the browser's benefit. The page posts
# the whole object back, and Config keeps unknown keys so that a file written
# by a newer build survives an older one -- which would otherwise mean these
# read-only extras get written into the user's config file. Strip them on the
# way in; they are outputs, not settings.
UI_ONLY_KEYS = ("path", "metric_specs")

# Preview frames are shrunk before encoding. The main stage wants a usable
# picture; a camera-manager thumbnail does not, and encoding a full frame five
# times a second for every camera would cost more than the pose inference does.
# The caller asks for a width and it is clamped here, so a stray query string
# cannot make the server encode something enormous.
PREVIEW_WIDTH = 640
PREVIEW_WIDTH_MIN = 96
PREVIEW_WIDTH_MAX = 960
PREVIEW_QUALITY = 72

# The panel is split into modules under static/. Only these extensions are
# served, and only from inside that directory -- see _static_file.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}


class ControlPanel:
    """Shared state the request handler acts on."""

    def __init__(self, monitor: Monitor, config_path: Path | None) -> None:
        self.monitor = monitor
        self.config_path = config_path
        self.shutdown_requested = threading.Event()
        self._lock = threading.Lock()

    # -- actions -----------------------------------------------------------

    def status(self) -> dict:
        snap = self.monitor.snapshot()
        snap["shutting_down"] = self.shutdown_requested.is_set()
        return snap

    def history(self, window_s: float | None = None) -> dict:
        history = getattr(self.monitor, "history", None)
        if history is None:
            return {"samples": [], "episodes": [], "session_average": None,
                    "count": 0, "now": 0.0, "window": 0.0, "interval": 0.0}
        return history.to_dict(window_s)

    def stats(self, day: str | None = None) -> dict:
        """One day's totals and timeline, straight from SQLite."""
        store = getattr(self.monitor, "store", None)
        if store is None:
            return {"available": False, "day": day,
                    "reason": "stats are switched off or the database "
                              "could not be opened"}
        try:
            payload = store.day_stats(day).to_dict()
        except ValueError:
            return {"available": False, "day": day, "reason": "not a valid date"}
        payload["available"] = True
        return payload

    def days(self) -> list[str]:
        store = getattr(self.monitor, "store", None)
        return [] if store is None else store.days()

    def persist(self, cfg: Config) -> None:
        """Write config the monitor changed itself, e.g. after calibrating."""
        try:
            cfg.save(self.config_path) if self.config_path else cfg.save()
        except OSError:
            log.exception("could not persist config after calibration")

    def config(self) -> dict:
        cfg = self.monitor.cfg
        data = cfg.to_dict()
        data["path"] = str(self.config_path) if self.config_path else None
        data["metric_specs"] = [
            {"key": s.key, "role": s.role, "label": s.label, "unit": s.unit,
             "description": s.description}
            for s in met.SPECS
        ]
        return data

    def preview_jpeg(self, camera: int, width: int = PREVIEW_WIDTH) -> bytes | None:
        """The newest frame for one camera, skeleton drawn on, as JPEG.

        Returns None when the preview is switched off or the camera has not
        produced a frame yet, which the handler turns into a 404 so the page
        can fall back to the stick figure.
        """
        if not retains_frames(self.monitor.cfg):
            return None
        latest = self.monitor.latest_preview(camera)
        if len(latest) != 3:
            return None
        image, landmarks, role = latest
        if image is None:
            return None

        # The frame is sent clean. The panel draws the posture geometry itself
        # in SVG, so drawing a second skeleton into the pixels here would stack
        # two overlays on one picture -- and a vector overlay can be styled,
        # scaled and coloured by posture state, which baked-in pixels cannot.
        # posture.overlay still serves the cv2 debug window.
        h, w = image.shape[:2]
        canvas = image
        target = max(PREVIEW_WIDTH_MIN, min(PREVIEW_WIDTH_MAX, int(width)))
        if w > target:
            canvas = cv2.resize(canvas, (target, max(1, int(h * target / w))),
                                interpolation=cv2.INTER_AREA)
        elif canvas is image:
            canvas = image.copy()
        ok, buf = cv2.imencode(".jpg", canvas,
                               [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_QUALITY])
        return buf.tobytes() if ok else None

    def devices(self, check: bool = False) -> dict:
        found = list_devices()
        if check:
            found.devices = [check_available(d.index) for d in found.devices]
        return found.to_dict()

    def save_config(self, payload: dict) -> dict:
        """Validate, persist and hot-apply a config edit."""
        with self._lock:
            payload = {k: v for k, v in payload.items() if k not in UI_ONLY_KEYS}
            cfg = Config.from_dict(payload)
            problems = validate(cfg)
            if problems:
                return {"ok": False, "problems": problems}
            if self.config_path:
                cfg.save(self.config_path)
            else:
                cfg.save()
            self.monitor.apply_config(cfg)
            return {"ok": True, "config": cfg.to_dict()}


def _query_float(path: str, key: str) -> float | None:
    """Pull one numeric query parameter, ignoring anything malformed.

    Parsed by hand rather than with ``urllib.parse``. That module cannot make a
    request, but the no-network invariant bans the whole ``urllib`` package by
    root name, and widening a security rule to save four lines is the wrong
    trade -- the rule is worth more than the convenience.
    """
    if "?" not in path:
        return None
    for pair in path.split("?", 1)[1].split("&"):
        name, _, raw = pair.partition("=")
        if name != key:
            continue
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _query_str(path: str, key: str) -> str | None:
    """Pull one string query parameter, restricted to safe characters.

    The only caller is the stats day, which is a date; anything else is
    rejected rather than passed to the store to interpret.
    """
    if "?" not in path:
        return None
    for pair in path.split("?", 1)[1].split("&"):
        name, _, raw = pair.partition("=")
        if name != key:
            continue
        if raw and all(c.isdigit() or c == "-" for c in raw) and len(raw) <= 10:
            return raw
        return None
    return None


def validate(cfg: Config) -> list[str]:
    """Reject settings that would produce a silently broken monitor."""
    problems: list[str] = []
    if cfg.web.host not in LOOPBACK:
        problems.append(
            f"Web interface host must stay on loopback, not {cfg.web.host!r}."
        )
    if not 1 <= cfg.web.port <= 65535:
        problems.append(f"Port {cfg.web.port} is out of range.")
    if not 0.2 <= cfg.sampling.fps <= 30:
        problems.append(f"Sample rate {cfg.sampling.fps} must be between 0.2 and 30 Hz.")
    if not 0.0 <= cfg.sampling.visibility_threshold <= 1.0:
        problems.append("Visibility threshold must be between 0 and 1.")
    a = cfg.alerts
    for name, value in (("subtle_after", a.subtle_after),
                        ("notify_after", a.notify_after),
                        ("overlay_after", a.overlay_after),
                        ("renotify_after", a.renotify_after)):
        if value < 0:
            problems.append(f"Alert timing {name} cannot be negative.")
    if not 0.5 <= a.clear_hold <= 120:
        problems.append("Hold-to-clear must be between 0.5 and 120 seconds.")
    if not 5 <= a.snooze_seconds <= 86400:
        problems.append("Snooze must be between 5 seconds and 24 hours.")
    if cfg.stats.retain_days < 0:
        problems.append("Stats retention cannot be negative.")
    if cfg.web.preview_mode not in PREVIEW_MODES:
        problems.append(
            f"Preview mode {cfg.web.preview_mode!r} must be one of "
            f"{', '.join(PREVIEW_MODES)}.")

    seen: set[int] = set()
    for cam in cfg.cameras:
        if cam.index in seen:
            problems.append(f"Camera index {cam.index} is listed more than once.")
        seen.add(cam.index)
        if cam.role not in ("side", "front"):
            problems.append(f"Camera {cam.index} has unknown role {cam.role!r}.")
        if cam.width <= 0 or cam.height <= 0:
            problems.append(f"Camera {cam.index} has an invalid resolution.")
    if not any(c.enabled for c in cfg.cameras):
        problems.append("At least one camera must be enabled.")
    return problems


class Handler(BaseHTTPRequestHandler):
    server_version = "posture"
    panel: ControlPanel  # set on the server instance

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        log.debug("http %s", fmt % args)

    # -- helpers -----------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The panel is loopback-only and generates everything itself; say so
        # explicitly so a browser extension cannot pull anything in either.
        self.send_header("Content-Security-Policy",
                         "default-src 'self' 'unsafe-inline'; img-src 'self' data:")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _guard(self) -> bool:
        """Refuse anything that did not come from this machine.

        The socket is bound to loopback so this should be unreachable, but a
        Host header check also blocks DNS-rebinding, where a page you visit
        resolves a name to 127.0.0.1 and talks to this server from your browser.
        """
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in LOOPBACK:
            self._json({"error": f"unexpected Host header {host!r}"}, 403)
            return False
        return True

    def _serve_static(self, relative: str) -> None:
        """Serve one file from the static directory, and nothing else.

        The path is resolved and then checked to be inside the directory, which
        rejects ``..`` traversal after normalisation rather than by pattern
        matching. Extensions are allowlisted so a stray file in there cannot be
        handed out as something the browser will execute differently.
        """
        try:
            target = (STATIC / relative).resolve()
            target.relative_to(STATIC.resolve())
        except (ValueError, OSError):
            self._json({"error": "not found"}, 404)
            return
        ctype = CONTENT_TYPES.get(target.suffix.lower())
        if ctype is None or not target.is_file():
            self._json({"error": "not found"}, 404)
            return
        self._send(200, target.read_bytes(), ctype)

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/history":
            window = _query_float(self.path, "window")
            self._json(self.panel.history(window))
        elif path == "/api/stats":
            self._json(self.panel.stats(_query_str(self.path, "day")))
        elif path == "/api/stats/days":
            self._json({"days": self.panel.days()})
        elif path == "/api/status":
            self._json(self.panel.status())
        elif path == "/api/config":
            self._json(self.panel.config())
        elif path == "/api/devices":
            self._json(self.panel.devices(check="check=1" in self.path))
        elif path.startswith("/api/frame/"):
            try:
                camera = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self._json({"error": "bad camera index"}, 400)
                return
            width = _query_float(self.path, "w") or PREVIEW_WIDTH
            body = self.panel.preview_jpeg(camera, int(width))
            if body is None:
                self._json({"error": "no preview available"}, 404)
                return
            self._send(200, body, "image/jpeg")
        elif path == "/api/metrics":
            self._json({"specs": [
                {"key": s.key, "role": s.role, "label": s.label, "unit": s.unit,
                 "direction": s.direction, "min_tolerance": s.min_tolerance,
                 "description": s.description}
                for s in met.SPECS
            ]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._json({"error": f"bad JSON: {exc}"}, 400)
            return

        if path == "/api/config":
            try:
                self._json(self.panel.save_config(payload))
            except Exception as exc:  # noqa: BLE001 - report, do not kill the server
                log.exception("config save failed")
                self._json({"ok": False, "problems": [str(exc)]}, 500)
        elif path == "/api/calibrate":
            camera = payload.get("camera")
            self._json(self.panel.monitor.start_calibration(
                None if camera is None else int(camera)))
        elif path == "/api/calibrate/cancel":
            self._json(self.panel.monitor.cancel_calibration())
        elif path == "/api/calibrate/clear":
            self._json(self.panel.monitor.clear_baselines())
        elif path == "/api/snooze":
            seconds = payload.get("seconds")
            self._json(self.panel.monitor.snooze(
                None if seconds is None else float(seconds)))
        elif path == "/api/snooze/cancel":
            self._json(self.panel.monitor.cancel_snooze())
        elif path == "/api/quit":
            self.panel.shutdown_requested.set()
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


class AlreadyRunning(OSError):
    """The port is taken -- almost always by another copy of this app."""

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        super().__init__(
            f"port {port} is already in use. Posture is probably already "
            f"running: open http://{host}:{port} instead. If it is not, "
            "change web.port in the config."
        )


class _Server(ThreadingHTTPServer):
    """A server that refuses to share its port on Windows.

    ``socketserver`` sets ``SO_REUSEADDR`` so a restart does not trip over a
    socket in TIME_WAIT. On Unix that is all it does. On Windows the same flag
    means something else entirely: a second process is allowed to bind a port
    that is already being listened on, and the two then split incoming
    connections between them at random.

    That is not theoretical. Two copies of this app ended up listening on 8760
    at once, both driving the cameras, both writing samples to the same SQLite
    file, and both showing a tray icon -- and a request to /api/stats was
    answered by whichever accepted first, so the panel showed data from a
    process nobody knew was running. Starting twice must fail loudly instead.
    """

    allow_reuse_address = os.name != "nt"


def serve(monitor: Monitor, web: WebConfig,
          config_path: Path | None = None) -> tuple[ThreadingHTTPServer, ControlPanel]:
    """Start the control panel. Refuses to bind anywhere but loopback."""
    if web.host not in LOOPBACK:
        raise ValueError(
            f"refusing to bind the control panel to {web.host!r}: "
            "this app is local-only and the interface must stay on loopback"
        )
    panel = ControlPanel(monitor, config_path)
    # A finished calibration changes config without anyone posting it, so the
    # monitor needs somewhere to write the new baselines.
    if hasattr(monitor, "set_config_change_hook"):
        monitor.set_config_change_hook(panel.persist)
    handler = type("BoundHandler", (Handler,), {"panel": panel})
    try:
        server = _Server((web.host, web.port), handler)
    except OSError as exc:
        raise AlreadyRunning(web.host, web.port) from exc
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="webui", daemon=True).start()
    return server, panel
