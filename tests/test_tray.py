"""The tray icon: colour and tooltip from a snapshot, and failing safely.

The rendering is trivial; what matters is the mapping from app state to the one
pixel of information a tray icon carries, and that a missing tray can never
take posture monitoring down with it.
"""

from __future__ import annotations

import pytest

from posture.tray import COLOURS, TrayIcon


def snapshot(**over):
    base = {
        "running": True,
        "posture": {"state": "good", "reason": ""},
        "score": {"value": 92, "band": "excellent"},
        "alert": {"level": 0, "snoozed": False, "headline": "",
                  "snooze_remaining": None},
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def read(**over):
    return TrayIcon()._read(snapshot(**over))  # noqa: SLF001


class TestColour:
    def test_good_posture_is_mint(self):
        tone, title, _ = read()
        assert tone == "good"
        assert "excellent" in title and "92" in title

    def test_a_fair_score_is_amber(self):
        tone, _t, _s = read(score={"value": 60, "band": "fair"})
        assert tone == "warn"

    def test_an_alert_turns_it_amber(self):
        tone, title, _ = read(alert={"level": 1, "headline": "Fix your neck tilt"})
        assert tone == "warn"
        assert "Fix your neck tilt" in title

    def test_the_overlay_step_turns_it_coral(self):
        tone, _t, _s = read(alert={"level": 3, "headline": "Fix your neck tilt"})
        assert tone == "bad"

    def test_the_alert_outranks_the_posture_state(self):
        """The thing asking for attention decides the colour."""
        tone, _t, _s = read(posture={"state": "good"},
                            alert={"level": 3, "headline": "Fix it"})
        assert tone == "bad"

    def test_an_empty_chair_is_grey_not_green(self):
        """Nobody there is not the same as sitting well."""
        tone, title, _ = read(posture={"state": "away"}, score={"value": None})
        assert tone == "idle"
        assert "nobody" in title

    def test_unreadable_landmarks_are_amber_and_say_why(self):
        tone, title, _ = read(posture={"state": "unknown",
                                       "reason": "cannot see left_hip"},
                              score={"value": None, "band": "no reading",
                                     "reason": "cannot see left_hip"})
        assert tone == "warn"
        assert "left_hip" in title

    def test_every_band_maps_to_a_tone(self):
        from posture.score import BANDS
        from posture.tray import BAND_TONE

        for _threshold, band, _tone in BANDS:
            assert band in BAND_TONE, band

    def test_slipping_before_an_alert_is_amber(self):
        tone, _t, _s = read(posture={"state": "bad"},
                            score={"value": 20, "band": "needs correction"})
        assert tone == "warn"

    def test_the_icon_never_contradicts_its_own_tooltip(self):
        """Mint with a tooltip reading "needs correction" is worse than either
        signal alone. Colour and words come from the same place.

        The detector is windowed and reports good until a deviation has been
        sustained; the score is near-instantaneous. Mixing them produced
        exactly that contradiction on real data.
        """
        tone, title, _ = read(posture={"state": "good"},
                              score={"value": 0, "band": "needs correction"})
        assert tone == "warn", "amber, because the tooltip says needs correction"
        assert "needs correction" in title

    def test_a_low_score_while_the_window_fills_is_not_mint(self):
        tone, _t, _s = read(posture={"state": "good", "reason": "still filling the window"},
                            score={"value": 30, "band": "needs correction"})
        assert tone == "warn"

    def test_not_monitoring_is_grey(self):
        tone, title, _ = read(running=False)
        assert tone == "idle" and "not monitoring" in title

    def test_snoozed_is_grey_and_counts_down(self):
        tone, title, snoozed = read(
            alert={"level": 0, "snoozed": True, "snooze_remaining": 605.0})
        assert tone == "idle" and snoozed is True
        assert "snoozed" in title and "10m" in title

    def test_snooze_outranks_an_active_alert(self):
        tone, _t, snoozed = read(
            alert={"level": 3, "snoozed": True, "snooze_remaining": 300.0,
                   "headline": "Fix it"})
        assert tone == "idle" and snoozed is True

    def test_every_tone_has_a_colour(self):
        for tone in ("good", "warn", "bad", "idle"):
            assert tone in COLOURS


class TestRendering:
    def test_a_glyph_is_produced_per_tone(self):
        pytest.importorskip("PIL")
        tray = TrayIcon()
        images = {tone: tray._image(tone) for tone in COLOURS}  # noqa: SLF001
        assert len({id(i) for i in images.values()}) == len(COLOURS)
        assert images["good"].size == images["bad"].size

    def test_glyphs_are_cached(self):
        pytest.importorskip("PIL")
        tray = TrayIcon()
        assert tray._image("good") is tray._image("good")  # noqa: SLF001

    def test_the_glyph_actually_uses_its_colour(self):
        pytest.importorskip("PIL")
        tray = TrayIcon()
        good = tray._image("good")  # noqa: SLF001
        # Sample the head rather than flattening every pixel: getdata() is
        # deprecated in Pillow 12 and this is a more direct assertion anyway.
        pixel = good.load()[32, 17]
        assert pixel[:3] == COLOURS["good"] and pixel[3] > 200
        assert good.load()[1, 1][3] == 0, "the background must stay transparent"


class TestFailingSafely:
    def test_update_before_start_is_a_no_op(self):
        """The app pushes snapshots twice a second whether or not a tray
        appeared; that must never raise."""
        TrayIcon().update(snapshot())

    def test_stop_without_start_is_a_no_op(self):
        TrayIcon().stop()

    def test_an_unavailable_tray_reports_false_and_does_not_start(self, monkeypatch):
        tray = TrayIcon()
        monkeypatch.setattr(tray, "_available", False)
        assert tray.available is False
        assert tray.start() is False

    def test_a_failing_backend_does_not_propagate(self, monkeypatch):
        """A missing tray icon must not stop posture monitoring."""

        tray = TrayIcon()
        monkeypatch.setattr(tray, "_menu", lambda: (_ for _ in ()).throw(RuntimeError))
        monkeypatch.setattr(tray, "_available", True)
        assert tray.start() is False
        assert tray.available is False

    def test_menu_actions_swallow_their_own_errors(self, monkeypatch):
        pytest.importorskip("pystray")

        def boom():
            raise RuntimeError("nope")

        tray = TrayIcon(on_open=boom)
        menu = tray._menu()  # noqa: SLF001
        items = list(menu)
        assert items, "the menu must have entries"
        # Invoking the action must not raise out of the tray thread.
        items[0](None)
