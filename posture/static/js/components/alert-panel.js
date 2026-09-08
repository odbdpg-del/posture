import { h, mount, fmt, shortDuration } from "../dom.js";

/**
 * What is wrong, and for how long.
 *
 * Driven by the detector's flagged metrics, which are already the product of a
 * 60-second window and hysteresis — so this cannot flicker on a moment's
 * movement, and it needs no animation to be noticed.
 */
export function AlertPanel(store) {
  const root = h("section", { class: "panel" },
    h("div", { class: "panel-head" }, "Current alerts"),
    h("div", { class: "panel-body" }, h("div", { class: "alerts" })));
  const list = root.querySelector(".alerts");

  function render() {
    const s = store.status || {};
    const posture = s.posture || {};
    const metrics = posture.metrics || [];
    const flagged = metrics.filter((m) => m.flagged);

    if (posture.state === "away") {
      mount(list, h("div", { class: "muted" }, "Monitoring paused — nobody at the desk."));
      return;
    }
    if (posture.state === "unknown") {
      mount(list, h("div", { class: "alert warn" },
        h("div", { class: "t" }, "Cannot read your posture"),
        h("div", { class: "d" }, posture.reason || "")));
      return;
    }
    if (!flagged.length) {
      mount(list, h("div", { class: "alert-ok" }, "✓ No active posture alerts"));
      // Near-threshold metrics are worth a quiet mention, but not an alert.
      const near = metrics.filter((m) => (m.ratio || 0) >= 0.5 && !m.flagged);
      if (near.length) {
        list.append(h("div", { class: "faint" },
          `Approaching threshold: ${near.map((m) => m.label).join(", ")}`));
      }
      return;
    }

    mount(list, flagged.map((m) => h("div", { class: "alert" },
      h("div", { class: "t" }, `${m.label} — ${fmt(m.excess, m.unit)} past baseline`),
      h("div", { class: "d" },
        `Duration ${shortDuration(s.state_since)} · out of tolerance ` +
        `${Math.round((m.out_fraction || 0) * 100)}% of the last minute`),
    )));
  }

  render();
  store.subscribe(render);
  return root;
}
