import { h, mount } from "../dom.js";

/**
 * Everything that used to crowd the main view: landmark confidences, backend
 * details, scale references, detector notes. Still one click away, but no
 * longer the first thing the app shows you.
 */
export function DiagnosticsPanel(store) {
  const body = h("div", { class: "panel-body" });
  const root = h("div", { class: "panel" },
    h("div", { class: "panel-head" }, "Diagnostics"), body);

  function render() {
    const s = store.status || {};
    const cams = s.cameras || [];
    const posture = s.posture || {};
    const thresh = store.config ? store.config.sampling.visibility_threshold : 0.6;

    mount(body,
      h("table", { class: "data" },
        h("tr", null, h("th", null, "Detector"), h("th", null, "")),
        row("State", `${posture.state || "—"}${posture.reason ? ` — ${posture.reason}` : ""}`),
        row("Calibrated", posture.calibrated ? "yes" : "no"),
        row("Worst ratio", (posture.worst_ratio ?? 0).toFixed(3)),
        row("Offenders", (posture.offenders || []).join(", ") || "none"),
        row("Session average", s.session_average ?? "—"),
        row("History samples", (store.history && store.history.count) || 0),
      ),
      ...cams.map((cam) => h("details", { class: "adv", open: cams.length === 1 },
        h("summary", null, `${cam.name} — landmarks and backend`),
        h("table", { class: "data" },
          row("State", cam.state_label),
          row("Backend", cam.backend || "—"),
          row("Resolution", cam.size ? cam.size.join(" × ") : "—"),
          row("Sample rate", cam.sample_hz ? `${cam.sample_hz} Hz` : "—"),
          row("Inference", cam.infer_ms ? `${cam.infer_ms} ms/frame` : "—"),
          row("Scale reference", cam.scale_kind || "—"),
          row("Near side", cam.near_side || "—"),
          row("Facing", cam.facing === 1 ? "+x" : cam.facing === -1 ? "−x" : "—"),
          row("Not in frame", (cam.missing || []).join(", ") || "none"),
        ),
        (cam.notes || []).length
          ? h("div", { class: "banner", style: { marginTop: "10px" } },
              h("ul", null, cam.notes.map((n) => h("li", null, n))))
          : null,
        h("div", { class: "vis-rows", style: { marginTop: "10px" } },
          Object.entries(cam.visibility || {}).map(([name, v]) =>
            h("div", { class: `vis-row ${v < thresh ? "low" : ""}` },
              h("span", null, name.replace(/_/g, " ")),
              h("span", { class: "num" }, v.toFixed(2)),
              h("span", { class: "bar" },
                h("i", { style: { width: `${Math.round(v * 100)}%` } })))))),
      ),
    );
  }

  function row(k, v) {
    return h("tr", null, h("td", { class: "muted" }, k),
             h("td", { class: "n num" }, String(v)));
  }

  render();
  store.subscribe(render);
  return root;
}
