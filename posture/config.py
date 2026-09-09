"""Configuration: dataclasses plus JSON load/save.

JSON rather than TOML because in phase 5 the web UI has to write this file back,
and the standard library reads TOML but cannot write it. Unknown keys are kept
and round-tripped so a config written by a newer build is not silently
destroyed by an older one.

Everything the app can be told to do lives here, and the panel writes it back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

Role = Literal["side", "front"]
PreviewMode = Literal["video", "skeleton", "off"]
PREVIEW_MODES = ("video", "skeleton", "off")


def retains_frames(cfg: "Config") -> bool:
    """Whether the workers should keep a frame in memory for the panel.

    The single place that answers it, so the capture side and the HTTP side
    cannot disagree about whether a frame should exist.
    """
    return cfg.web.preview_mode == "video"

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".posture" / "config.json"


@dataclass
class CameraConfig:
    """One physical camera and the role it plays.

    ``role`` drives which metrics get computed. Everything works with a single
    camera in either role, just with fewer metrics available.
    """

    index: int = 0
    role: Role = "side"
    name: str = ""
    enabled: bool = True
    width: int = 640
    height: int = 480
    backend: str | None = None  # None lets us try MSMF, then DSHOW, then ANY

    def label(self) -> str:
        return self.name or f"cam{self.index} ({self.role})"


@dataclass
class SamplingConfig:
    """How often we look, and how sure we have to be about a landmark.

    ``fps`` is the peak rate, not a constant one. Pose inference costs about
    13 ms of CPU per frame and running it five times a second all day came to
    roughly 21% of a core -- over budget for something meant to sit in the
    background. Since the rate only has to be high when a decision is close,
    it follows the detector: full rate when a metric is near its tolerance or
    posture is already bad, ``idle_fps`` when everything is comfortably fine,
    and ``away_fps`` at an empty chair.
    """

    fps: float = 5.0
    adaptive: bool = True
    idle_fps: float = 2.0
    away_fps: float = 0.5
    # Above this fraction of a metric's tolerance, go back to full rate. Well
    # below 1.0 so the rate rises before the verdict does, not after.
    active_ratio: float = 0.5
    # Landmarks below this visibility are treated as unreadable rather than
    # trusted. MediaPipe reports poorly-supported joints with high confidence
    # surprisingly often, so this sits above the model's own gate.
    visibility_threshold: float = 0.6
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class DetectionConfig:
    """When a run of bad readings becomes a bad posture. See detector.py."""

    window_seconds: float = 60.0
    bad_fraction: float = 0.70
    exit_ratio: float = 0.70
    absence_seconds: float = 15.0
    tolerance_multiplier: float = 3.0
    min_window_fill: float = 0.30
    # Confidence a metric needs before it may raise an alert. Below this it is
    # still measured and still shown, faded; it just does not interrupt you.
    min_confidence: float = 0.75
    # Per-metric tolerance overrides, by metric key. Empty means "derive it
    # from my calibration", which is the intended way round.
    overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationConfig:
    """How the baseline capture behaves."""

    duration: float = 10.0
    min_samples: int = 15


@dataclass
class AlertsConfig:
    """Escalation timings and the snooze. See alerts.py for the mechanics."""

    enabled: bool = True
    # Seconds of sustained bad posture before each step. Cumulative, so the
    # defaults mean colour at once, a notification at 30 s, the overlay at 90 s.
    subtle_after: float = 0.0
    notify_after: float = 30.0
    overlay_after: float = 60.0
    # Continuous verified-good seconds needed to dismiss the overlay.
    clear_hold: float = 5.0
    # Seconds the view of you may be lost before a part-completed hold restarts.
    hold_grace: float = 1.5
    snooze_seconds: float = 600.0
    renotify_after: float = 0.0
    # The two effects can be turned off independently of the escalation, for
    # anyone who wants the panel to know without the machine interrupting.
    os_notifications: bool = True
    fullscreen_overlay: bool = True


@dataclass
class StatsConfig:
    """Durable stats in local SQLite. Derived numbers only, never an image."""

    enabled: bool = True
    # None means ~/.posture/posture.db, next to this config file.
    path: str | None = None
    # Rows older than this are pruned at startup. Ninety days of samples is a
    # few tens of megabytes and more history than anyone reads.
    retain_days: int = 90


@dataclass
class WebConfig:
    """The local control panel.

    ``host`` is not meant to be changed. It is a field rather than a constant
    only so the value is visible in the config file; binding anywhere other
    than loopback would expose a live camera feed's derived data to the
    network, and ``webui.serve`` refuses to do it.
    """

    host: str = "127.0.0.1"
    port: int = 8760
    open_browser: bool = True
    # The tray icon is the app's home once the browser tab is closed, and the
    # first step of the alert escalation. Off means the app only exists while
    # the panel is open.
    tray_icon: bool = True
    # What the panel shows in the camera workspace:
    #
    #   "video"     the camera image, with posture geometry drawn over it
    #   "skeleton"  posture geometry only, on a blank panel
    #   "off"       neither
    #
    # Only "video" causes a frame to be retained or encoded anywhere. Skeleton
    # mode draws from the landmark coordinates already in the status payload,
    # so choosing it is a real reduction in what leaves the process, not a
    # cosmetic one -- and it is the setting for anyone who does not want to
    # watch themselves on screen all day.
    preview_mode: str = "video"
    # Kept for configs written before preview_mode existed. from_dict migrates
    # it and stops writing it; nothing reads it at runtime.
    video_preview: bool = True


@dataclass
class Config:
    version: int = 1
    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    web: WebConfig = field(default_factory=WebConfig)
    # Calibrated baselines, keyed by camera index as a string because JSON
    # object keys are strings. Derived numbers only -- centres, spreads and
    # sample counts -- which is all this app is ever allowed to persist.
    baselines: dict[str, Any] = field(default_factory=dict)
    model_path: str | None = None  # None means the bundled default location
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self._extra)
        out.update({
            "version": self.version,
            "cameras": [asdict(c) for c in self.cameras],
            "sampling": asdict(self.sampling),
            "detection": asdict(self.detection),
            "calibration": asdict(self.calibration),
            "alerts": asdict(self.alerts),
            "stats": asdict(self.stats),
            "web": {k: v for k, v in asdict(self.web).items()
                    if k != "video_preview"},
            "baselines": self.baselines,
            "model_path": self.model_path,
        })
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {"version", "cameras", "sampling", "detection", "calibration",
                 "alerts", "stats", "web", "baselines", "model_path"}
        cams = [_build(CameraConfig, c) for c in data.get("cameras", [])] or [CameraConfig()]
        return cls(
            version=int(data.get("version", 1)),
            cameras=cams,
            sampling=_build(SamplingConfig, data.get("sampling", {})),
            detection=_build(DetectionConfig, data.get("detection", {})),
            calibration=_build(CalibrationConfig, data.get("calibration", {})),
            alerts=_build(AlertsConfig, data.get("alerts", {})),
            stats=_build(StatsConfig, data.get("stats", {})),
            web=_migrate_web(data.get("web", {})),
            baselines=dict(data.get("baselines") or {}),
            model_path=data.get("model_path"),
            _extra={k: v for k, v in data.items() if k not in known},
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config, falling back to defaults when the file is absent."""
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            return cls()
        cfg = cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        # A file can be hand-edited, and an unusable value in it should not stop
        # the app from starting. Repair it and say so.
        if cfg.web.preview_mode not in PREVIEW_MODES:
            log.warning("unknown preview_mode %r in %s; using 'video'",
                        cfg.web.preview_mode, p)
            cfg.web.preview_mode = "video"
            cfg.web.video_preview = True
        return cfg

    def save(self, path: str | Path | None = None) -> Path:
        """Write config atomically, so a crash mid-write cannot truncate it."""
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(p)
        return p


def _migrate_web(data: dict[str, Any] | None) -> WebConfig:
    """Build WebConfig, honouring the older boolean if that is all there is.

    ``video_preview: false`` meant "do not show me the camera image", which is
    what skeleton mode now means, so an old config keeps behaving the way its
    owner chose rather than silently switching the camera back on.
    """
    data = dict(data or {})
    if "preview_mode" not in data and "video_preview" in data:
        data["preview_mode"] = "video" if data.get("video_preview") else "skeleton"
    web = _build(WebConfig, data)
    # Deliberately not coerced here. from_dict is also how a submitted config
    # arrives, and quietly repairing a bad value there would mean validate()
    # never sees it and the user is never told their setting was ignored.
    # Config.load repairs it instead, for a hand-edited file.
    web.video_preview = web.preview_mode == "video"
    return web


def _build(kind: type, data: dict[str, Any] | None):
    """Construct a dataclass from a dict, ignoring keys it does not know."""
    data = data or {}
    names = {f.name for f in fields(kind)}
    return kind(**{k: v for k, v in data.items() if k in names})
