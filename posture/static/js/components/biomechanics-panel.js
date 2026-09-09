import { h, mount, reconcile, toneForRatio } from "../dom.js";

/**
 * Every metric as live / baseline / difference, with what it is worth.
 *
 * The score answers "how are you sitting"; this answers "what is actually
 * different, and can that be trusted". They are separate questions and the
 * second one is the one you act on, so the difference column is the point of
 * the table — the live and baseline columns are there to make it accountable.
 *
 * Precision follows the measurement rather than the float. A tenth of a degree
 * from a landmark the model is unsure of is decoration, so the digits shown
 * depend on confidence; nothing here ever prints 23.4817°.
 *
 * A metric that cannot be measured is a row saying why, not a blank and never
 * a zero. On a desk camera the hips are usually out of frame, so trunk lean is
 * routinely unavailable — showing 0° for it would be a fabricated reading of
 * the one metric people most expect to see.
 */
export function BiomechanicsPanel(store) {
  const body = h("div", { class: "panel-body biomech" });
  const summary = h("div", { class: "biomech-summary" });
  const root = h("section", { class: "panel" },
    h("div", { class: "panel-head" },
      h("span", null, "Biomechanics"), h("span", { style: { flex: "1" } }), summary),
    body);

  function render() {
    const status = store.status;
    if (!status) return;
    const posture = status.posture || {};
    const rows = posture.metrics || [];
    const suspect = new Set(posture.suspect || []);
    const cameras = status.cameras || [];

    // Metrics this setup can produce at all, versus ones it is measuring right
    // now. The gap between them is the useful part: "not being measured" is a
    // fact about your camera placement, and hiding it makes the table look
    // like the whole picture when it is not.
    const measurable = new Set(rows.map((m) => m.key));
    const absent = [];
    for (const cam of cameras) {
      for (const key of ROLE_METRICS[cam.role] || []) {
        if (!measurable.has(key)) absent.push([key, cam]);
      }
    }

    const sig = [rows.map((m) => m.key).join(","),
                 absent.map(([k]) => k).join(","),
                 [...suspect].join(",")].join("|");
    reconcile(body, sig, () => build(rows, absent, suspect));
    update(rows, posture, cameras);
  }

  const cells = new Map();

  function build(rows, absent, suspect) {
    cells.clear();
    const table = h("table", { class: "data biomech-table" },
      h("tr", null,
        h("th", null, "Metric"),
        h("th", { class: "n" }, "Live"),
        h("th", { class: "n" }, "Baseline"),
        h("th", { class: "n" }, "Difference"),
        h("th", { class: "n" }, "Confidence")));

    for (const m of rows) {
      const live = h("td", { class: "n num" });
      const base = h("td", { class: "n num faint" });
      const diff = h("td", { class: "n num" });
      const conf = h("td", { class: "n num" });
      const label = h("td", null,
        h("span", { class: "bm-label" }, m.label),
        suspect.has(m.label) ? h("span", { class: "bm-note bad" }, "baseline unreachable") : null);
      cells.set(m.key, { live, base, diff, conf });
      table.append(h("tr", { class: "bm-row" }, label, live, base, diff, conf));
    }

    for (const [key, cam] of absent) {
      table.append(h("tr", { class: "bm-row absent" },
        h("td", null,
          h("span", { class: "bm-label" }, METRIC_LABELS[key] || key),
          h("span", { class: "bm-note" }, reasonFor(cam))),
        h("td", { class: "n num faint", colspan: "4" }, "not measurable")));
    }
    return table;
  }

  function update(rows, posture, cameras) {
    for (const m of rows) {
      const c = cells.get(m.key);
      if (!c) continue;
      const low = m.low_confidence;
      const digits = decimals(m);
      c.live.textContent = num(m.value, m.unit, digits);
      c.live.className = `n num ${low ? "faded" : ""}`;
      c.base.textContent = num(m.baseline, m.unit, digits);
      const d = (m.value !== null && m.baseline !== null
                 && m.value !== undefined && m.baseline !== undefined)
        ? m.value - m.baseline : null;
      c.diff.textContent = d === null ? "—" : signed(d, m.unit, digits);
      c.diff.className = `n num ${low ? "faded" : toneForRatio(m.ratio)}`;
      c.conf.textContent = m.confidence === null || m.confidence === undefined
        ? "—" : `${Math.round(m.confidence * 100)}%`;
      c.conf.className = `n num ${low ? "bad" : "faint"}`;
      c.conf.title = low
        ? "Below the confidence needed to raise an alert. Measured, shown, not acted on."
        : "";
    }

    const tracked = cameras.filter((c) => c.state === "tracking").length;
    const worst = rows.reduce((a, m) => Math.max(a, m.ratio || 0), 0);
    mount(summary,
      h("span", { class: "faint" },
        `${rows.length} measured · ${tracked}/${cameras.length} cameras tracking`),
      posture.state
        ? h("span", { class: `tag ${toneForRatio(worst)}` }, STATE_WORD[posture.state]
            || posture.state)
        : null);
  }

  render();
  store.subscribe(render);
  return root;
}

const STATE_WORD = { good: "within baseline", bad: "sustained deviation",
                     unknown: "not measurable", away: "nobody at the desk" };

const METRIC_LABELS = {
  neck_tilt: "Neck tilt", neck_flexion: "Neck flexion", forward_head: "Forward head",
  torso_lean: "Trunk lean", shoulder_tilt: "Shoulder slope", head_roll: "Head tilt",
  lateral_offset: "Lateral offset",
};

const ROLE_METRICS = {
  side: ["neck_tilt", "forward_head", "neck_flexion", "torso_lean"],
  front: ["shoulder_tilt", "head_roll", "lateral_offset"],
};

/** Why a metric this camera should produce is not arriving. */
function reasonFor(cam) {
  const missing = cam.missing || [];
  if (missing.length) return `${cam.name}: cannot see ${missing.join(", ").replace(/_/g, " ")}`;
  const note = (cam.notes || [])[0];
  if (note) return `${cam.name}: ${note}`;
  if (cam.state === "no_person") return `${cam.name}: nobody in view`;
  return `${cam.name}: not calibrated`;
}

/**
 * How many decimals this reading has earned.
 *
 * Confidence decides it: a landmark the model half-guessed does not support a
 * tenth of a degree, and printing one anyway claims a precision the
 * measurement does not have.
 */
function decimals(m) {
  if (m.unit !== "deg") return 2;
  return (m.confidence ?? 1) >= 0.9 ? 1 : 0;
}

function unitSuffix(unit) {
  return unit === "deg" ? "°" : unit === "x torso" || unit === "x shoulders" ? "×" : "";
}

function num(value, unit, digits) {
  if (value === null || value === undefined) return "—";
  return `${Number(value).toFixed(digits)}${unitSuffix(unit)}`;
}

function signed(value, unit, digits) {
  const n = Number(value);
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}${unitSuffix(unit)}`;
}
