"""The system tray icon: step one of the escalation, and the way back in.

The brief's first alert step is "a tray icon colour change", and that is what
this is -- the only part of the escalation that costs you nothing to ignore. It
also gives the app somewhere to live once the browser tab is closed, which
until now it did not have.

Colour is the same vocabulary as the panel: mint for good, amber for a
deviation worth noticing, coral once the overlay is up, grey for anything the
app cannot or should not judge. Nothing animates; the icon is glanceable, not
attention-seeking.

Everything here degrades to nothing. If pystray or Pillow are missing, or the
platform has no tray, :attr:`TrayIcon.available` is False and the rest of the
app carries on -- a missing tray icon must never stop posture monitoring.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

# Matches the panel's palette so the two never disagree about what amber means.
COLOURS = {
    "good": (74, 222, 155),
    "warn": (232, 179, 74),
    "bad": (240, 106, 93),
    "idle": (110, 125, 118),
}

# An active alert always decides the colour: it is the thing asking for
# attention. Otherwise the colour comes from the *score band*, and so does the
# tooltip text.
#
# Using the detector state for the colour and the score for the words made the
# icon contradict itself -- mint, with a tooltip reading "needs correction (0)".
# They answer different questions: the detector is deliberately windowed and
# says "good" until a deviation has been sustained, while the score is a
# mostly-instantaneous read. A tray icon carries one pixel of information and
# cannot afford to disagree with its own label, so both now come from the same
# place. The score is already damped by the out-of-tolerance fraction, so it
# does not flicker.
BAND_TONE = {
    "excellent": "good",
    "good": "good",
    "fair": "warn",
    "needs correction": "warn",
}
# Only reached when there is no score at all -- away, unreadable, or before
# calibration.
STATE_TONE = {
    "good": "good",
    "bad": "warn",
    "unknown": "warn",
    "away": "idle",
}
LEVEL_TONE = {1: "warn", 2: "warn", 3: "bad"}

ICON_SIZE = 64


def _glyph(colour: tuple[int, int, int]):
    """A seated figure, drawn once per colour.

    Kept to two shapes because the icon is often rendered at 16 pixels, where
    anything more detailed turns to mush.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = colour + (255,)
    # Head.
    draw.ellipse((23, 8, 41, 26), fill=fill)
    # Shoulders and torso, tapering, so the silhouette reads as a person
    # rather than a blob at small sizes.
    draw.polygon([(16, 56), (20, 34), (44, 34), (48, 56)], fill=fill)
    return image


class TrayIcon:
    """A tray icon that follows posture state, with a small menu."""

    def __init__(self, *, on_open=None, on_snooze=None, on_resume=None,
                 on_calibrate=None, on_quit=None) -> None:
        self._on_open = on_open
        self._on_snooze = on_snooze
        self._on_resume = on_resume
        self._on_calibrate = on_calibrate
        self._on_quit = on_quit

        self._icon = None
        self._images: dict[str, object] = {}
        self._tone = None
        self._title = ""
        self._snoozed = False
        self._available: bool | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import pystray  # noqa: F401
                from PIL import Image  # noqa: F401
                self._available = True
            except Exception:
                log.info("no system tray available (pystray/Pillow missing)")
                self._available = False
        return self._available

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Show the icon. Returns whether it actually appeared."""
        if not self.available or self._icon is not None:
            return self._icon is not None
        import pystray

        try:
            self._icon = pystray.Icon(
                "posture", icon=self._image("idle"), title="Posture — starting",
                menu=self._menu(),
            )
            # Detached: the tray runs its own message loop on its own thread,
            # leaving the main thread free to own shutdown.
            self._icon.run_detached()
        except Exception:
            log.warning("could not start the tray icon", exc_info=True)
            self._icon = None
            self._available = False
            return False
        return True

    def stop(self) -> None:
        icon, self._icon = self._icon, None
        if icon is None:
            return
        try:
            icon.stop()
        except Exception:  # pragma: no cover - teardown best effort
            log.debug("tray stop failed", exc_info=True)

    # -- state -------------------------------------------------------------

    def update(self, status: dict) -> None:
        """Push the latest snapshot in. Cheap enough to call twice a second."""
        if self._icon is None:
            return
        tone, title, snoozed = self._read(status)
        with self._lock:
            changed_menu = snoozed != self._snoozed
            self._snoozed = snoozed
            if tone != self._tone:
                self._tone = tone
                try:
                    self._icon.icon = self._image(tone)
                except Exception:
                    log.debug("tray icon update failed", exc_info=True)
            if title != self._title:
                self._title = title
                try:
                    # Windows truncates tooltips around 128 characters.
                    self._icon.title = title[:127]
                except Exception:
                    log.debug("tray tooltip update failed", exc_info=True)
        if changed_menu:
            try:
                self._icon.update_menu()
            except Exception:
                log.debug("tray menu update failed", exc_info=True)

    def _read(self, status: dict) -> tuple[str, str, bool]:
        alert = status.get("alert") or {}
        posture = status.get("posture") or {}
        score = status.get("score") or {}
        snoozed = bool(alert.get("snoozed"))

        if not status.get("running"):
            return "idle", "Posture — not monitoring", snoozed
        if snoozed:
            left = alert.get("snooze_remaining")
            mins = "" if left is None else f" ({int(left // 60)}m left)"
            return "idle", f"Posture — snoozed{mins}", snoozed

        level = int(alert.get("level") or 0)
        if level:
            tone = LEVEL_TONE.get(level, "warn")
            headline = alert.get("headline") or "Posture needs correcting"
            return tone, f"Posture — {headline}", snoozed

        state = posture.get("state") or "unknown"
        if state == "away":
            return "idle", "Posture — nobody at the desk", snoozed

        value = score.get("value")
        if value is not None:
            band = score.get("band") or "unknown"
            return BAND_TONE.get(band, "warn"), f"Posture — {band} ({value})", snoozed

        # No score: say why rather than guessing a colour for it.
        reason = score.get("reason") or posture.get("reason") or "no reading"
        return STATE_TONE.get(state, "idle"), f"Posture — {reason}", snoozed

    def _image(self, tone: str):
        if tone not in self._images:
            self._images[tone] = _glyph(COLOURS.get(tone, COLOURS["idle"]))
        return self._images[tone]

    # -- menu --------------------------------------------------------------

    def _menu(self):
        import pystray

        def call(fn):
            def handler(icon=None, item=None):
                if fn is None:
                    return
                try:
                    fn()
                except Exception:
                    log.exception("tray action failed")
            return handler

        return pystray.Menu(
            pystray.MenuItem("Open panel", call(self._on_open), default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Snooze alerts", call(self._on_snooze),
                             visible=lambda item: not self._snoozed),
            pystray.MenuItem("Resume alerts", call(self._on_resume),
                             visible=lambda item: self._snoozed),
            pystray.MenuItem("Calibrate", call(self._on_calibrate)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Stop monitoring", call(self._on_quit)),
        )
