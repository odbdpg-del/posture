import { h, mount } from "../dom.js";
import { api } from "../api.js";

// Which baselines a role can produce, for the checklist. Mirrors the server's
// SPECS_BY_ROLE; the labels come from the live payload so they cannot drift.
const ROLE_METRICS = {
  side: ["neck_tilt", "neck_flexion", "forward_head", "torso_lean"],
  front: ["shoulder_tilt", "head_roll", "lateral_offset"],
};

/**
 * Guided baseline capture.
 *
 * Same server-side calibration as before — start, watch progress, collect the
 * result — presented as a short sequence instead of a dense diagnostic block.
 * The raw numbers are still one disclosure away.
 */
/** Whether any stored baseline carries the neutral pose the ghost overlay
 *  draws. Baselines captured before it was recorded have none. */
function hasPose(s) {
  return Object.values((s && s.baselines) || {})
    .some((b) => ((b || {}).landmarks || []).length > 0);
}

export function CalibrationWizard(store) {
  const root = h("div", { class: "panel" },
    h("div", { class: "panel-head" }, "Calibration"),
    h("div", { class: "panel-body" }));
  const body = root.lastChild;
  let busy = false;

  async function begin() {
    busy = true; render();
    try { await api.calibrate(null); } finally { busy = false; }
  }

  async function clearAll() {
    if (!confirm("Clear your calibrated baselines? Posture scoring stops until "
                 + "you calibrate again.")) return;
    await api.clearBaselines();
  }

  function labelFor(key) {
    const specs = (store.config && store.config.metric_specs) || [];
    const found = specs.find((s) => s.key === key);
    return found ? found.label : key.replace(/_/g, " ");
  }

  function render() {
    const s = store.status || {};
    const calibrating = (s.cameras || []).find((c) => c.calibrating);
    const result = s.calibration_result;
    const calibrated = s.posture && s.posture.calibrated;

    if (calibrating) { renderCapture(calibrating); return; }
    if (result) { renderResult(result); return; }
    renderIdle(calibrated, s);
  }

  function renderIdle(calibrated, s) {
    const uncal = s.uncalibrated_metrics || [];
    // Metrics the detector has stopped judging because their baseline cannot
    // be satisfied by any normal posture. The wizard is where this belongs:
    // the only fix is to calibrate again, and the button is right here.
    const suspect = (s.posture && s.posture.suspect) || [];
    mount(body, h("div", { class: "wizard" },
      h("div", { class: "step" }, calibrated ? "Calibrated" : "Step 1 of 1"),
      h("h3", null, calibrated ? "Baseline is set" : "Set your neutral posture"),
      h("p", null, calibrated
        ? "Everything is measured against how you sat during calibration. "
          + "Recalibrate if you change chair, desk or camera position."
        : "Sit naturally in your preferred neutral working position and hold "
          + "still. It watches for ten seconds, takes the median of every "
          + "frame so a fidget cannot move the result, and remembers that as "
          + "your baseline — nothing is compared against a textbook ideal."),
      suspect.length ? h("div", { class: "banner bad" },
        `Not being judged: ${suspect.join(", ")}. `
        + "The baseline was captured somewhere a normal posture cannot reach, so "
        + "sitting well would never clear it. Recalibrate sitting the way you "
        + "actually want to sit.") : null,
      calibrated && !hasPose(s) ? h("div", { class: "banner info" },
        "This baseline has no stored pose, so the overlay's baseline comparison "
        + "has nothing to draw. It was captured before the app recorded one; "
        + "recalibrating adds it.") : null,
      uncal.length ? h("div", { class: "banner info" },
        `Measurable but not calibrated: ${uncal.join(", ")}. `
        + "Recalibrate to start using them.") : null,
      h("div", { class: "row", style: { justifyContent: "center" } },
        h("button", { class: "primary", disabled: busy || !s.running, onClick: begin },
          calibrated ? "Recalibrate neutral posture" : "Calibrate neutral posture"),
        calibrated ? h("button", { class: "danger", onClick: clearAll },
          "Clear baseline") : null),
      !s.running ? h("p", { class: "faint", style: { marginTop: "14px" } },
        "No camera is running.") : null,
    ));
  }

  function renderCapture(cam) {
    const cal = cam.calibrating;
    const remaining = Math.max(0, cal.duration - cal.elapsed);
    const counts = cal.counts || {};
    const keys = ROLE_METRICS[cam.role] || [];
    mount(body, h("div", { class: "wizard" },
      h("div", { class: "step" }, "Capturing"),
      h("h3", null, "Hold still"),
      h("p", null, "Sit naturally and face the way you normally work."),
      h("div", { class: "countdown num" }, String(Math.ceil(remaining))),
      h("div", { class: "calprogress" },
        h("i", { style: { width: `${Math.round(cal.progress * 100)}%` } })),
      h("div", { class: "checklist" }, keys.map((k) => h("div", {
        class: `check ${counts[k] ? "done" : ""}`,
      }, h("span", { class: "box" }, counts[k] ? "✓" : "○"),
         `${labelFor(k)} baseline`,
         counts[k] ? h("span", { class: "faint" }, ` ${counts[k]} samples`) : null))),
      h("button", { class: "ghost", onClick: () => api.cancelCalibration() }, "Cancel"),
    ));
  }

  function renderResult(result) {
    const ok = result.ok;
    mount(body, h("div", { class: "wizard" },
      h("div", { class: "step" }, ok ? "Done" : "Failed"),
      h("h3", null, ok ? "Baseline saved" : "Calibration failed"),
      h("p", null, ok
        ? "Posture is now measured against this. Recalibrate any time."
        : "No baseline was recorded. Sit in view of the camera for the full ten "
          + "seconds and try again."),
      h("div", { class: "checklist" }, (result.cameras || []).flatMap((c) => {
        const keys = ROLE_METRICS[c.role] || [];
        return keys.map((k) => h("div", {
          class: `check ${c.metrics[k] ? "done" : "miss"}`,
        }, h("span", { class: "box" }, c.metrics[k] ? "✓" : "—"),
           `${labelFor(k)} — ${c.name}`));
      })),
      (result.problems || []).length
        ? h("div", { class: "banner bad", style: { textAlign: "left" } },
            h("div", null, "Notes"),
            h("ul", null, result.problems.map((p) => h("li", null, p))))
        : null,
      h("div", { class: "row", style: { justifyContent: "center" } },
        h("button", { class: "primary", disabled: busy, onClick: begin },
          "Calibrate again")),
      h("details", { class: "adv", style: { textAlign: "left" } },
        h("summary", null, "Advanced calibration data"),
        h("table", { class: "data" },
          h("tr", null, ["Camera", "Metric", "Centre", "Spread", "Samples"]
            .map((t) => h("th", null, t))),
          ...(result.cameras || []).flatMap((c) =>
            Object.entries(c.metrics || {}).map(([k, m]) => h("tr", null,
              h("td", null, c.name), h("td", null, labelFor(k)),
              h("td", { class: "n num" }, m.centre.toFixed(2)),
              h("td", { class: "n num" }, `±${m.spread.toFixed(2)}`),
              h("td", { class: "n num" }, String(m.samples))))))),
    ));
  }

  render();
  store.subscribe(render);
  return root;
}
