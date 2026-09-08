import { h, toneForRatio, STATUS_WORD } from "../dom.js";

// A metric's bar runs 0..1.5x tolerance so there is visible travel past the
// limit; the limit tick sits at two thirds.
const SCALE = 1.5;

/**
 * One compact instrument: value, position against tolerance, and a status word.
 * Small on purpose — these are readouts, not cards.
 */
export function MetricGauge(metric) {
  const ratio = metric.ratio;
  const tone = metric.value === null || metric.value === undefined
    ? "idle" : toneForRatio(ratio);
  const pct = Math.min(100, ((ratio || 0) / SCALE) * 100);

  return h("div", {
    class: `gauge ${tone}`,
    title: metric.baseline === null || metric.baseline === undefined ? ""
      : `baseline ${metric.baseline} · tolerance ${metric.tolerance} ${metric.unit}`,
  },
    h("div", { class: "top" },
      h("span", { class: "name" }, metric.label),
      h("span", { class: "val num" }, ...valueParts(metric)),
    ),
    h("div", { class: "track" },
      h("i", { class: "fill", style: { width: `${pct}%` } }),
      h("b", { class: "limit" }),
    ),
    h("div", { class: "foot" },
      h("span", { class: `status ${tone}` }, STATUS_WORD[tone]),
      h("span", { class: "num" },
        metric.out_fraction ? `${Math.round(metric.out_fraction * 100)}% out` : ""),
    ),
  );
}

// Degrees get a tight symbol; ratio units ("x torso") are too long to sit at
// full size next to the number, so they are set small and dim.
function valueParts(metric) {
  if (metric.value === null || metric.value === undefined) return ["—"];
  const n = Number(metric.value);
  const text = `${n > 0 ? "+" : ""}${n.toFixed(metric.unit === "deg" ? 1 : 2)}`;
  if (metric.unit === "deg") return [`${text}°`];
  return [text, h("span", { class: "u" }, metric.unit)];
}

/** The horizontal strip of gauges under the camera. */
export function MetricStrip(store) {
  const root = h("section", { class: "panel strip" },
    h("div", { class: "panel-head" }, "Measurements"),
    h("div", { class: "panel-body" }));
  const body = root.lastChild;

  function render() {
    const posture = (store.status && store.status.posture) || {};
    const metrics = posture.metrics || [];
    body.replaceChildren(...(metrics.length
      ? metrics.map(MetricGauge)
      : [h("div", { class: "muted", style: { padding: "14px" } },
          posture.calibrated === false
            ? "Calibrate to start measuring against your baseline."
            : "No measurements available right now.")]));
  }

  render();
  store.subscribe(render);
  return root;
}
