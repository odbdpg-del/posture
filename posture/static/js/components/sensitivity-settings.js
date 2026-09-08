import { h, mount } from "../dom.js";
import { api } from "../api.js";
import { refreshConfig } from "../store.js";

const SOURCE_TEXT = {
  learned: (m) => `learned from you (3 × ${(m.derived / 3).toFixed(2)} spread)`,
  floored: (m) => `floor ${m.floor} — your ${m.derived.toFixed(1)} was tighter`,
  capped: (m) => `ceiling ${m.ceiling} — your ${m.derived.toFixed(1)} was wider`,
  manual: () => "set by you",
};

/**
 * Per-metric tolerances, with provenance.
 *
 * Same detection.overrides the server already honours — an empty box means
 * "use my calibration", a number bypasses the clamps. Lives under Settings
 * because it is a tuning surface, not something to read while working.
 */
export function SensitivitySettings(store) {
  const rows = h("div");
  const banner = h("div");
  const dirty = h("span", { class: "faint" });
  const applyBtn = h("button", { class: "primary", onClick: apply }, "Apply sensitivity");

  const root = h("div", { class: "panel" },
    h("div", { class: "panel-head" }, "Sensitivity"),
    h("div", { class: "panel-body" },
      h("p", { class: "muted", style: { marginTop: "0" } },
        "How far a metric may stray from your baseline before it counts as bad. "
        + "Smaller means it complains sooner."),
      banner, rows,
      h("div", { class: "row", style: { marginTop: "12px" } },
        applyBtn,
        h("button", { onClick: clearAll }, "Clear all overrides"),
        dirty)));

  function collect() {
    const out = {};
    rows.querySelectorAll("input[data-key]").forEach((input) => {
      const raw = input.value.trim();
      if (raw === "") return;
      const value = parseFloat(raw);
      if (!Number.isNaN(value)) out[input.dataset.key] = value;
    });
    return out;
  }

  async function apply() {
    const cfg = JSON.parse(JSON.stringify(store.config));
    cfg.detection.overrides = collect();
    applyBtn.disabled = true;
    try {
      const res = await api.saveConfig(cfg);
      mount(banner, h("div", { class: `banner ${res.ok ? "ok" : "bad"}` },
        res.ok ? "Saved." : "Not saved",
        res.ok ? null : h("ul", null, (res.problems || []).map((p) => h("li", null, p)))));
      if (res.ok) { await refreshConfig(); dirty.textContent = ""; }
      if (res.ok) setTimeout(() => mount(banner), 3500);
    } finally { applyBtn.disabled = false; }
  }

  async function clearAll() {
    rows.querySelectorAll("input[data-key]").forEach((i) => { i.value = ""; });
    await apply();
  }

  function render() {
    const metrics = ((store.status && store.status.posture) || {}).metrics || [];
    const overrides = (store.config && store.config.detection.overrides) || {};
    if (!metrics.length) {
      mount(rows, h("div", { class: "muted" },
        "Calibrate first — there is nothing to be sensitive about yet."));
      return;
    }
    const key = metrics.map((m) => m.key).join(",");
    if (rows.dataset.key !== key) {
      rows.dataset.key = key;
      mount(rows, h("table", { class: "data" },
        h("tr", null, ["Metric", "In use", "Where it came from", "Override"]
          .map((t) => h("th", null, t))),
        ...metrics.map((m) => h("tr", { dataset: { row: m.key } },
          h("td", null, m.label),
          h("td", { class: "n num", dataset: { cell: "inuse" } }, ""),
          h("td", { class: "muted", dataset: { cell: "src" } }, ""),
          h("td", null, h("input", {
            type: "number", step: "0.01", min: "0", placeholder: "auto",
            dataset: { key: m.key }, style: { width: "6rem" },
            onInput: () => { dirty.textContent = "unsaved changes"; },
          }))))));
    }
    for (const m of metrics) {
      const row = rows.querySelector(`[data-row="${m.key}"]`);
      if (!row) continue;
      row.querySelector('[data-cell="inuse"]').textContent = `${m.tolerance} ${m.unit}`;
      const fn = SOURCE_TEXT[m.source] || (() => m.source);
      row.querySelector('[data-cell="src"]').textContent = fn(m);
      const input = row.querySelector("input");
      if (document.activeElement !== input) {
        const v = overrides[m.key];
        input.value = v === undefined || v === null ? "" : v;
      }
    }
  }

  render();
  store.subscribe(render);
  return root;
}
