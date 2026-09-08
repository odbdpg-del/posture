import { h, mount } from "../dom.js";
import { api } from "../api.js";
import { refreshConfig } from "../store.js";

import { TopStatusBar } from "./top-status-bar.js";
import { Sidebar } from "./sidebar.js";
import { LiveCameraView } from "./live-camera-view.js";
import { PostureScore } from "./posture-score.js";
import { AlertPanel } from "./alert-panel.js";
import { AlertBanner } from "./alert-banner.js";
import { MetricStrip } from "./metric-gauge.js";
import { PostureTimeline } from "./posture-timeline.js";
import { DayStats } from "./day-stats.js";
import { CalibrationWizard } from "./calibration-wizard.js";
import { CameraManager } from "./camera-manager.js";
import { SensitivitySettings } from "./sensitivity-settings.js";
import { DiagnosticsPanel } from "./diagnostics-panel.js";

/**
 * Layout and routing only.
 *
 * Views are built once and swapped, not rebuilt on every poll — each component
 * subscribes to the store and updates itself. That keeps the 2 Hz status poll
 * from tearing down DOM the user is interacting with (a half-typed override,
 * an open dropdown) and keeps this file about arrangement rather than data.
 */
export function AppShell(store) {
  const view = h("main", { class: "view" });
  const shell = h("div", { class: "shell" },
    TopStatusBar(store, quit),
    h("div", { class: "body" }, Sidebar(store, navigate), view));

  const views = {};
  let current = null;

  function build(name) {
    if (views[name]) return views[name];
    views[name] = {
      live: buildLive,
      history: buildHistory,
      cameras: buildCameras,
      calibration: buildCalibration,
      settings: buildSettings,
    }[name](store);
    return views[name];
  }

  function navigate(name) {
    if (store.view === name) return;
    store.set({ view: name });
  }

  async function quit() {
    if (!confirm("Stop monitoring and release the cameras?")) return;
    await api.quit();
  }

  function render() {
    shell.classList.toggle("collapsed", store.sidebarCollapsed);
    if (current !== store.view) {
      current = store.view;
      mount(view, build(current));
    }
  }

  render();
  store.subscribe(render);
  return shell;
}

/* ── views ─────────────────────────────────────────────────────────────── */

function buildLive(store) {
  return h("div", { class: "live" },
    LiveCameraView(store),
    h("div", { class: "side" },
      PostureScore(store), AlertBanner(store), AlertPanel(store),
      PostureTimeline(store, { compact: true })),
    MetricStrip(store));
}

function buildHistory(store) {
  return h("div", { class: "stack" },
    DayStats(store),
    PostureTimeline(store),
    EpisodeList(store));
}

function buildCameras(store) {
  return h("div", { class: "stack" }, CameraManager(store));
}

function buildCalibration(store) {
  return h("div", { class: "stack" }, CalibrationWizard(store));
}

function buildSettings(store) {
  return h("div", { class: "stack" },
    MonitoringSettings(store),
    AlertSettings(store),
    SensitivitySettings(store),
    DiagnosticsPanel(store));
}

/* ── small view-local pieces ───────────────────────────────────────────── */

function EpisodeList(store) {
  const body = h("div", { class: "panel-body" });
  const root = h("div", { class: "panel" },
    h("div", { class: "panel-head" }, "Posture events"), body);

  function render() {
    const episodes = ((store.history && store.history.episodes) || []).slice().reverse();
    if (!episodes.length) {
      mount(body, h("div", { class: "muted" },
        "No posture events recorded yet. Sustained deviations appear here."));
      return;
    }
    mount(body, h("table", { class: "data" },
      h("tr", null, ["Time", "Duration", "Metrics", "Low score"]
        .map((t) => h("th", null, t))),
      ...episodes.map((ep) => h("tr", null,
        h("td", { class: "num" },
          new Date(ep.started * 1000).toLocaleTimeString([],
            { hour: "numeric", minute: "2-digit" })),
        h("td", { class: "num" }, `${Math.round(ep.duration)}s`),
        h("td", null, (ep.offenders || []).map((o) => o.replace(/_/g, " ")).join(", ")),
        h("td", { class: "n num" }, String(ep.worst_score))))));
  }

  render();
  store.subscribe(render);
  return root;
}

function AlertSettings(store) {
  const banner = h("div");
  const fields = h("div", { class: "row" });
  const root = h("div", { class: "panel" },
    h("div", { class: "panel-head" }, "Alerts"),
    h("div", { class: "panel-body" },
      h("p", { class: "muted", style: { marginTop: "0" } },
        "Escalation is cumulative from the moment posture goes bad. The overlay "
        + "closes only after you have held a good posture for the hold time — "
        + "looking away does not count."),
      banner, fields));

  const num = (id, label, value, attrs = {}) => h("div", { class: "field" },
    h("label", { for: id }, label),
    h("input", { id, type: "number", min: "0", step: "1", value, ...attrs }));

  async function save() {
    const cfg = JSON.parse(JSON.stringify(store.config));
    const a = cfg.alerts;
    a.enabled = fields.querySelector("#al_on").checked;
    a.notify_after = parseFloat(fields.querySelector("#al_notify").value);
    a.overlay_after = parseFloat(fields.querySelector("#al_overlay").value);
    a.clear_hold = parseFloat(fields.querySelector("#al_hold").value);
    a.snooze_seconds = parseFloat(fields.querySelector("#al_snooze").value);
    a.os_notifications = fields.querySelector("#al_os").checked;
    a.fullscreen_overlay = fields.querySelector("#al_full").checked;
    const res = await api.saveConfig(cfg);
    mount(banner, h("div", { class: `banner ${res.ok ? "ok" : "bad"}` },
      res.ok ? "Saved." : "Not saved",
      res.ok ? null : h("ul", null, (res.problems || []).map((p) => h("li", null, p)))));
    if (res.ok) { await refreshConfig(); setTimeout(() => mount(banner), 3500); }
  }

  let built = false;
  function render() {
    if (!store.config || built) return;
    built = true;
    const a = store.config.alerts;
    mount(fields,
      h("div", { class: "field" }, h("label", { for: "al_on" }, "Alerts on"),
        h("input", { id: "al_on", type: "checkbox", checked: a.enabled })),
      num("al_notify", "Notify after (s bad)", a.notify_after),
      num("al_overlay", "Overlay after (s more)", a.overlay_after),
      num("al_hold", "Hold to clear (s)", a.clear_hold, { min: "0.5", step: "0.5" }),
      num("al_snooze", "Snooze length (s)", a.snooze_seconds, { min: "5" }),
      h("div", { class: "field" }, h("label", { for: "al_os" }, "OS notification"),
        h("input", { id: "al_os", type: "checkbox", checked: a.os_notifications })),
      h("div", { class: "field" }, h("label", { for: "al_full" }, "Full-screen overlay"),
        h("input", { id: "al_full", type: "checkbox", checked: a.fullscreen_overlay })),
      h("div", { class: "field" }, h("label", null, " "),
        h("button", { class: "primary", onClick: save }, "Save alerts")));
  }

  render();
  store.subscribe(render);
  return root;
}

function MonitoringSettings(store) {
  const banner = h("div");
  const fields = h("div", { class: "row" });
  const root = h("div", { class: "panel" },
    h("div", { class: "panel-head" }, "Monitoring"),
    h("div", { class: "panel-body" }, banner, fields));

  const input = (id, label, attrs) => h("div", { class: "field" },
    h("label", { for: id }, label), h("input", { id, ...attrs }));

  async function save() {
    const cfg = JSON.parse(JSON.stringify(store.config));
    cfg.sampling.fps = parseFloat(fields.querySelector("#fps").value);
    cfg.sampling.visibility_threshold =
      parseFloat(fields.querySelector("#vis").value);
    cfg.sampling.adaptive = fields.querySelector("#adaptive").checked;
    cfg.web.port = parseInt(fields.querySelector("#port").value, 10);
    cfg.web.open_browser = fields.querySelector("#openb").checked;
    cfg.web.tray_icon = fields.querySelector("#tray").checked;
    cfg.web.preview_mode = fields.querySelector("#preview").value;
    const res = await api.saveConfig(cfg);
    mount(banner, h("div", { class: `banner ${res.ok ? "ok" : "bad"}` },
      res.ok ? "Saved. Cameras restarting…" : "Not saved",
      res.ok ? null : h("ul", null, (res.problems || []).map((p) => h("li", null, p)))));
    if (res.ok) { await refreshConfig(); setTimeout(() => mount(banner), 4000); }
  }

  let built = false;
  function render() {
    if (!store.config || built) return;
    built = true;
    const c = store.config;
    mount(fields,
      input("fps", "Peak sample rate (Hz)",
        { type: "number", min: "0.2", max: "30", step: "0.5", value: c.sampling.fps }),
      input("vis", "Visibility threshold",
        { type: "number", min: "0", max: "1", step: "0.05",
          value: c.sampling.visibility_threshold }),
      input("port", "Panel port",
        { type: "number", min: "1", max: "65535", value: c.web.port }),
      h("div", { class: "field" }, h("label", { for: "adaptive" }, "Adaptive rate"),
        h("input", { id: "adaptive", type: "checkbox", checked: c.sampling.adaptive })),
      h("div", { class: "field", style: { minWidth: "190px" } },
        h("label", { for: "preview" }, "Camera preview"),
        h("select", { id: "preview" },
          h("option", { value: "video", selected: c.web.preview_mode === "video" },
            "Camera video"),
          h("option", { value: "skeleton", selected: c.web.preview_mode === "skeleton" },
            "Skeleton only"),
          h("option", { value: "off", selected: c.web.preview_mode === "off" },
            "Off"))),
      h("div", { class: "field" }, h("label", { for: "openb" }, "Open browser on start"),
        h("input", { id: "openb", type: "checkbox", checked: c.web.open_browser })),
      h("div", { class: "field" }, h("label", { for: "tray" }, "Tray icon"),
        h("input", { id: "tray", type: "checkbox", checked: c.web.tray_icon })),
      h("div", { class: "field" }, h("label", null, " "),
        h("button", { class: "primary", onClick: save }, "Save and apply")),
    );
    root.append(h("p", { class: "faint", style: { padding: "0 12px 12px" } },
      "Skeleton only draws the tracked figure and no camera image — nothing "
      + "is kept in memory or encoded in that mode, so it is a real reduction "
      + "in what leaves the process, not just a different picture. "
      + "Port changes take effect on next start; everything else applies "
      + "immediately."));
  }

  render();
  store.subscribe(render);
  return root;
}
