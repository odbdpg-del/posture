"""Control panel: validation, loopback binding, and the JSON API.

These run against a real server on a real socket, but never touch a camera --
the monitor is stubbed. The point is the request handling and the settings
rules, which are what break when config editing is added to.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from posture.config import CameraConfig, Config, WebConfig
from posture.history import History
from posture.webui import AlreadyRunning, serve, validate


class FakeMonitor:
    """Stands in for Monitor without opening anything."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.applied: list[Config] = []
        self.history = History()
        self.snoozed: list = []

    def snapshot(self) -> dict:
        return {"uptime": 1.0, "cameras": [], "roles": [], "any_tracking": False,
                "running": False, "configured_count": len(self.cfg.cameras)}

    def apply_config(self, cfg: Config) -> None:
        self.applied.append(cfg)
        self.cfg = cfg

    def snooze(self, seconds=None) -> dict:
        self.snoozed.append(seconds)
        return {"snoozed": True, "snooze_remaining": seconds or 600}

    def cancel_snooze(self) -> dict:
        self.snoozed.append("cancel")
        return {"snoozed": False}


@pytest.fixture
def panel_server(tmp_path):
    cfg = Config()
    cfg.web = WebConfig(host="127.0.0.1", port=0, open_browser=False)
    monitor = FakeMonitor(cfg)
    server, panel = serve(monitor, cfg.web, tmp_path / "config.json")
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, monitor, panel, tmp_path / "config.json"
    finally:
        server.shutdown()
        server.server_close()


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return json.loads(response.read())


def expect_error(base: str, path: str, headers: dict | None = None) -> int:
    """Request something expected to fail, and return the status code.

    Closes the error response explicitly: an HTTPError *is* a response object,
    and letting it be finalised by the garbage collector raises
    "I/O operation on closed file" from the interpreter shutdown path, which
    pytest surfaces as an unraisable-exception warning.
    """
    request = urllib.request.Request(base + path, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
            return exc.code
        finally:
            exc.close()


def post(base: str, path: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestBinding:
    def test_refuses_to_bind_off_loopback(self):
        """A config edit must not be able to put this on the network."""
        with pytest.raises(ValueError, match="loopback"):
            serve(FakeMonitor(Config()), WebConfig(host="0.0.0.0", port=0))

    def test_a_second_copy_is_refused_not_silently_allowed(self, tmp_path):
        """Two of these must never listen on one port.

        socketserver sets SO_REUSEADDR, which on Windows lets a second process
        bind a port that is already being listened on. Two copies then ran at
        once -- both driving the cameras, both writing to the same database --
        and requests were answered by whichever accepted first.
        """
        cfg = Config()
        cfg.web.port = 0
        first, _panel = serve(FakeMonitor(cfg), cfg.web, tmp_path / "a.json")
        try:
            taken = WebConfig(host="127.0.0.1", port=first.server_address[1])
            with pytest.raises(AlreadyRunning) as caught:
                serve(FakeMonitor(Config()), taken, tmp_path / "b.json")
            assert "already running" in str(caught.value)
        finally:
            first.shutdown()
            first.server_close()

    def test_already_running_is_an_oserror(self):
        """The CLI catches OSError around bind; this must not slip past it."""
        assert issubclass(AlreadyRunning, OSError)

    def test_binds_loopback_only(self, panel_server):
        base, *_ = panel_server
        assert get(base, "/api/status")["running"] is False

    def test_rejects_foreign_host_header(self, panel_server):
        """Blocks DNS rebinding: a remote page resolving a name to 127.0.0.1."""
        base, *_ = panel_server
        assert expect_error(base, "/api/status",
                            {"Host": "evil.example.com"}) == 403


class TestApi:
    def test_serves_the_page(self, panel_server):
        base, *_ = panel_server
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            body = response.read().decode()
        assert "<title>Posture</title>" in body
        assert response.headers["Content-Security-Policy"]

    def test_config_round_trip(self, panel_server):
        base, monitor, _, path = panel_server
        cfg = get(base, "/api/config")
        cfg["cameras"] = [
            {"index": 0, "role": "side", "name": "A", "enabled": True,
             "width": 640, "height": 480, "backend": None},
            {"index": 1, "role": "front", "name": "B", "enabled": True,
             "width": 640, "height": 480, "backend": None},
        ]
        cfg["sampling"]["fps"] = 3.0
        # The fixture binds an ephemeral port; 0 is fine to listen on but not a
        # meaningful thing to save, and validate() rightly refuses it.
        cfg["web"]["port"] = 8765
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"], body
        assert path.exists()
        saved = json.loads(path.read_text())
        assert len(saved["cameras"]) == 2
        assert saved["sampling"]["fps"] == 3.0
        assert monitor.applied, "a saved config must be applied to the monitor"

    def test_saved_config_has_no_ui_only_keys(self, panel_server):
        """The page posts back what /api/config gave it, extras included.

        Config deliberately preserves unknown keys so a file from a newer build
        survives an older one; without stripping, that helpfulness would write
        the whole metric catalogue into the user's settings file every save.
        """
        base, _, _, path = panel_server
        cfg = get(base, "/api/config")
        assert "metric_specs" in cfg and "path" in cfg, "fixture assumes these exist"
        cfg["web"]["port"] = 8765
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"], body
        saved = json.loads(path.read_text())
        assert "metric_specs" not in saved
        assert "path" not in saved

    def test_config_exposes_metric_descriptions(self, panel_server):
        base, *_ = panel_server
        keys = {m["key"] for m in get(base, "/api/config")["metric_specs"]}
        assert {"neck_flexion", "shoulder_tilt"} <= keys

    def test_bad_config_is_rejected_without_applying(self, panel_server):
        base, monitor, _, path = panel_server
        cfg = get(base, "/api/config")
        cfg["sampling"]["fps"] = 999
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"] is False
        assert any("Sample rate" in p for p in body["problems"])
        assert not monitor.applied
        assert not path.exists(), "a rejected config must not be written"

    def test_malformed_json_is_a_400(self, panel_server):
        base, *_ = panel_server
        request = urllib.request.Request(
            base + "/api/config", data=b"{not json",
            headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 400
        exc.value.read(); exc.value.close()

    def test_quit_sets_the_shutdown_flag(self, panel_server):
        base, _, panel, _ = panel_server
        assert not panel.shutdown_requested.is_set()
        status, body = post(base, "/api/quit", {})
        assert status == 200 and body["ok"]
        assert panel.shutdown_requested.is_set()

    def test_unknown_route_is_a_404(self, panel_server):
        base, *_ = panel_server
        assert expect_error(base, "/api/nope") == 404


class TestValidate:
    def base(self, **kw) -> Config:
        cfg = Config()
        cfg.web = WebConfig(host="127.0.0.1", port=8760)
        for key, value in kw.items():
            setattr(cfg, key, value)
        return cfg

    def test_default_config_is_valid(self):
        assert validate(self.base()) == []

    def test_rejects_non_loopback_host(self):
        cfg = self.base()
        cfg.web.host = "0.0.0.0"
        assert any("loopback" in p for p in validate(cfg))

    def test_rejects_duplicate_camera_index(self):
        cfg = self.base(cameras=[CameraConfig(index=1, role="side"),
                                 CameraConfig(index=1, role="front")])
        assert any("more than once" in p for p in validate(cfg))

    def test_rejects_all_cameras_disabled(self):
        cfg = self.base(cameras=[CameraConfig(index=0, enabled=False)])
        assert any("At least one camera" in p for p in validate(cfg))

    def test_rejects_unknown_role(self):
        cfg = self.base(cameras=[CameraConfig(index=0, role="diagonal")])
        assert any("unknown role" in p for p in validate(cfg))

    @pytest.mark.parametrize("fps", [0.0, 0.1, 31.0])
    def test_rejects_absurd_sample_rates(self, fps):
        cfg = self.base()
        cfg.sampling.fps = fps
        assert any("Sample rate" in p for p in validate(cfg))

    def test_accepts_two_cameras_in_different_roles(self):
        cfg = self.base(cameras=[CameraConfig(index=0, role="side"),
                                 CameraConfig(index=1, role="front")])
        assert validate(cfg) == []


class TestToleranceOverrides:
    """The sensitivity control writes detection.overrides and hot-applies it."""

    def test_override_round_trips_and_is_applied(self, panel_server):
        base, monitor, _, path = panel_server
        cfg = get(base, "/api/config")
        cfg["web"]["port"] = 8765
        cfg["detection"]["overrides"] = {"neck_flexion": 12.0}
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"], body
        saved = json.loads(path.read_text())
        assert saved["detection"]["overrides"] == {"neck_flexion": 12.0}
        assert monitor.applied[-1].detection.overrides == {"neck_flexion": 12.0}

    def test_clearing_overrides_returns_to_the_learned_value(self, panel_server):
        base, monitor, _, path = panel_server
        cfg = get(base, "/api/config")
        cfg["web"]["port"] = 8765
        cfg["detection"]["overrides"] = {"neck_flexion": 12.0}
        post(base, "/api/config", cfg)
        cfg["detection"]["overrides"] = {}
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"]
        assert json.loads(path.read_text())["detection"]["overrides"] == {}


class TestStaticAssets:
    """The panel is split into modules, so the server has to serve them."""

    def test_serves_css_and_js_with_the_right_types(self, panel_server):
        base, *_ = panel_server
        for path, expected in (("/static/css/app.css", "text/css"),
                               ("/static/js/app.js", "text/javascript"),
                               ("/static/js/components/app-shell.js", "text/javascript")):
            with urllib.request.urlopen(base + path, timeout=5) as response:
                assert response.status == 200, path
                assert expected in response.headers["Content-Type"], path
                assert response.read(), path

    def test_refuses_to_climb_out_of_the_static_directory(self, panel_server):
        """Resolve-then-contain, so `..` is rejected after normalisation."""
        base, *_ = panel_server
        for attempt in ("/static/../webui.py", "/static/../../README.md",
                        "/static/..%2fwebui.py"):
            assert expect_error(base, attempt) == 404, attempt

    def test_refuses_extensions_that_are_not_panel_assets(self, panel_server, tmp_path):
        base, *_ = panel_server
        assert expect_error(base, "/static/js/app.js.map") == 404

    def test_missing_file_is_a_404_not_a_500(self, panel_server):
        base, *_ = panel_server
        assert expect_error(base, "/static/js/nope.js") == 404


class TestHistoryEndpoint:
    def test_a_monitor_without_history_still_answers(self):
        """Older or stubbed monitors must not 500 the timeline."""
        from posture.webui import ControlPanel

        class NoHistory:
            cfg = Config()
            def snapshot(self): return {}

        body = ControlPanel(NoHistory(), None).history()
        assert body["samples"] == [] and body["count"] == 0

    def test_history_is_served_even_with_no_data(self, panel_server):
        base, *_ = panel_server
        body = get(base, "/api/history")
        assert set(body) >= {"samples", "episodes", "session_average", "count"}
        assert body["samples"] == [] and body["episodes"] == []

    def test_window_parameter_is_accepted(self, panel_server):
        base, *_ = panel_server
        assert get(base, "/api/history?window=600")["window"] == 600

    def test_a_malformed_window_does_not_break_it(self, panel_server):
        base, *_ = panel_server
        assert "samples" in get(base, "/api/history?window=abc")


class TestPreviewModes:
    """Skeleton and off must genuinely stop frames, not just hide them."""

    def test_default_is_video(self, panel_server):
        base, *_ = panel_server
        assert get(base, "/api/config")["web"]["preview_mode"] == "video"

    def test_mode_round_trips(self, panel_server):
        base, monitor, _, path = panel_server
        cfg = get(base, "/api/config")
        cfg["web"]["port"] = 8765
        cfg["web"]["preview_mode"] = "skeleton"
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"], body
        assert json.loads(path.read_text())["web"]["preview_mode"] == "skeleton"

    def test_an_unknown_mode_is_rejected(self, panel_server):
        base, *_ = panel_server
        cfg = get(base, "/api/config")
        cfg["web"]["port"] = 8765
        cfg["web"]["preview_mode"] = "hologram"
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"] is False
        assert any("Preview mode" in p for p in body["problems"])

    @pytest.mark.parametrize("mode", ["skeleton", "off"])
    def test_no_frame_is_served_outside_video_mode(self, panel_server, mode):
        base, monitor, panel, _ = panel_server
        monitor.cfg.web.preview_mode = mode
        assert panel.preview_jpeg(0) is None
        assert expect_error(base, "/api/frame/0") == 404

    def test_the_legacy_boolean_is_migrated(self):
        """A config written before preview_mode existed must keep behaving the
        way its owner chose, not silently switch the camera back on."""
        from posture.config import Config

        assert Config.from_dict({"web": {"video_preview": False}}) \
            .web.preview_mode == "skeleton"
        assert Config.from_dict({"web": {"video_preview": True}}) \
            .web.preview_mode == "video"

    def test_the_panel_never_reads_the_retired_flag(self):
        """The old boolean is no longer serialised, so any JS still testing it
        reads undefined and silently means "always on" — which made the camera
        manager request frames in skeleton mode and 404 continuously."""
        import pathlib

        import posture

        retired = "video" + "_preview"
        static = pathlib.Path(posture.__file__).resolve().parent / "static"
        for path in list(static.rglob("*.js")) + list(static.rglob("*.html")):
            assert retired not in path.read_text(encoding="utf-8"), path.name

    def test_the_legacy_key_is_not_written_back(self):
        """Keeping both would let them drift apart."""
        from posture.config import Config

        assert "video_preview" not in Config().to_dict()["web"]

    def test_a_hand_edited_file_is_repaired_on_load(self, tmp_path):
        """Strict at the API, forgiving of a file someone edited by hand."""
        from posture.config import Config

        path = tmp_path / "cfg.json"
        cfg = Config()
        data = cfg.to_dict()
        data["web"]["preview_mode"] = "hologram"
        path.write_text(json.dumps(data))
        assert Config.load(path).web.preview_mode == "video"


class TestSnoozeApi:
    def test_snooze_uses_the_configured_length_by_default(self, panel_server):
        base, monitor, _, _ = panel_server
        status, body = post(base, "/api/snooze", {})
        assert status == 200 and body["snoozed"] is True
        assert monitor.snoozed == [None]

    def test_an_explicit_length_is_passed_through(self, panel_server):
        base, monitor, _, _ = panel_server
        post(base, "/api/snooze", {"seconds": 120})
        assert monitor.snoozed == [120.0]

    def test_snooze_can_be_cancelled(self, panel_server):
        base, monitor, _, _ = panel_server
        status, body = post(base, "/api/snooze/cancel", {})
        assert status == 200 and body["snoozed"] is False
        assert monitor.snoozed == ["cancel"]


class TestAlertValidation:
    def base(self) -> Config:
        cfg = Config()
        cfg.web = WebConfig(host="127.0.0.1", port=8760)
        return cfg

    def test_defaults_are_valid(self):
        assert validate(self.base()) == []

    def test_negative_timings_are_rejected(self):
        cfg = self.base()
        cfg.alerts.notify_after = -5
        assert any("cannot be negative" in p for p in validate(cfg))

    @pytest.mark.parametrize("hold", [0.0, 0.2, 500.0])
    def test_absurd_hold_times_are_rejected(self, hold):
        cfg = self.base()
        cfg.alerts.clear_hold = hold
        assert any("Hold-to-clear" in p for p in validate(cfg))

    @pytest.mark.parametrize("snooze", [1.0, 200000.0])
    def test_absurd_snooze_lengths_are_rejected(self, snooze):
        cfg = self.base()
        cfg.alerts.snooze_seconds = snooze
        assert any("Snooze" in p for p in validate(cfg))

    def test_alert_settings_round_trip(self, panel_server):
        base, _, _, path = panel_server
        cfg = get(base, "/api/config")
        cfg["web"]["port"] = 8765
        cfg["alerts"]["overlay_after"] = 45.0
        cfg["alerts"]["clear_hold"] = 8.0
        status, body = post(base, "/api/config", cfg)
        assert status == 200 and body["ok"], body
        saved = json.loads(path.read_text())["alerts"]
        assert saved["overlay_after"] == 45.0 and saved["clear_hold"] == 8.0


class TestStatsApi:
    def test_days_and_stats_are_served(self, panel_server, tmp_path):
        base, monitor, _, _ = panel_server
        from posture.store import Store, day_bounds

        monitor.store = Store(tmp_path / "stats.db")
        start, _end = day_bounds("2026-03-04")
        for i in range(30):
            monitor.store.add_sample(start + 9 * 3600 + i * 2, 90, "good")
        try:
            assert "2026-03-04" in get(base, "/api/stats/days")["days"]
            stats = get(base, "/api/stats?day=2026-03-04")
            assert stats["available"] is True
            assert stats["good_percent"] == 100.0
            assert stats["samples"] == 30
            assert stats["timeline"]
        finally:
            monitor.store.close()

    def test_stats_say_so_when_there_is_no_database(self, panel_server):
        base, monitor, _, _ = panel_server
        monitor.store = None
        body = get(base, "/api/stats")
        assert body["available"] is False and body["reason"]
        assert get(base, "/api/stats/days")["days"] == []

    def test_a_malformed_day_is_rejected_not_passed_through(self, panel_server):
        """The day goes into a date parser; only date-shaped input gets there."""
        base, monitor, _, _ = panel_server
        from posture.store import Store

        monitor.store = Store(":memory:")
        try:
            body = get(base, "/api/stats?day=../../etc/passwd")
            # Rejected at the query parser, so it falls back to today.
            assert body["available"] is True
            assert body["day"] != "../../etc/passwd"
        finally:
            monitor.store.close()

    def test_retention_cannot_be_negative(self):
        cfg = Config()
        cfg.web = WebConfig(host="127.0.0.1", port=8760)
        cfg.stats.retain_days = -1
        assert any("retention" in p for p in validate(cfg))
