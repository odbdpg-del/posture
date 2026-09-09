// Biomechanical overlay: the geometry the app actually measures, drawn over
// the feed with the angles it derived from it.
//
// Coordinate system
// -----------------
// The viewBox is `0 0 (100*aspect) 100`, which is `geometry.to_metric_frame`
// scaled by 100 with y pointing down. Two things follow, and both matter:
// one SVG unit is the same length horizontally and vertically, so arcs stay
// circular and text is not stretched; and an angle drawn here is the same
// angle the server measured, so an arc cannot disagree with the number
// printed beside it.
//
// A square viewBox stretched over a 4:3 frame — which is what this was before
// — gets the landmark positions right and everything derived from them wrong.
//
// Nothing here computes a posture number. Angles come from the server, which
// took them from aspect-corrected, smoothed landmarks; the arcs are drawn from
// the same landmark positions so that the picture and the reading are two
// views of one measurement rather than two independent guesses.

import { toneForRatio } from "../dom.js";

// Landmark order as sent by the server (monitor.PREVIEW_LANDMARKS).
const P = {
  nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6, leftElbow: 7, rightElbow: 8,
  leftHip: 9, rightHip: 10, leftKnee: 11, rightKnee: 12,
};

const SVG_NS = "http://www.w3.org/2000/svg";

// What each mode draws. Ordered from least to most, and every mode is a
// superset of the one before it except "minimal", which is the opt-out.
export const MODES = [
  ["minimal", "Minimal"],
  ["landmarks", "Landmarks"],
  ["angles", "Angles"],
  ["baseline", "Baseline comparison"],
  ["full", "Full biomechanics"],
];

const SHOWS = {
  minimal: {},
  landmarks: { joints: true },
  angles: { joints: true, geometry: true, arcs: true, refs: true },
  baseline: { joints: true, geometry: true, arcs: true, refs: true, ghost: true },
  full: { joints: true, geometry: true, arcs: true, refs: true, ghost: true,
          limbs: true, secondary: true },
};

const DEG = 180 / Math.PI;

export function PostureOverlay() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("preserveAspectRatio", "none");

  function el(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined) continue;
      node.setAttribute(k, v);
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /**
   * Arc between two directions about a vertex, taking the short way round.
   *
   * Angles are screen-space radians (y down). The short way is always the one
   * meant: these are deviations from a reference axis, and a neck tilted 20°
   * off vertical is never the 340° explementary angle.
   */
  function arcPath(vx, vy, from, to, r) {
    let delta = to - from;
    while (delta <= -Math.PI) delta += 2 * Math.PI;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    const x0 = vx + r * Math.cos(from), y0 = vy + r * Math.sin(from);
    const x1 = vx + r * Math.cos(from + delta), y1 = vy + r * Math.sin(from + delta);
    return {
      d: `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 0 `
         + `${delta > 0 ? 1 : 0} ${x1.toFixed(2)} ${y1.toFixed(2)}`,
      mid: from + delta / 2,
      delta,
    };
  }

  /**
   * @param cam      camera snapshot (landmarks, size, role)
   * @param posture  verdict, for per-metric value/ratio/confidence
   * @param thresh   visibility threshold, so we hide what the app itself ignores
   * @param opts     { mode, baseline } — baseline is the stored neutral pose
   */
  function render(cam, posture, thresh, opts = {}) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const mode = opts.mode || "angles";
    const show = SHOWS[mode] || SHOWS.angles;
    const lm = (cam && cam.landmarks) || [];

    const size = (cam && cam.size) || null;
    const aspect = size && size[1] ? size[0] / size[1] : 4 / 3;
    const W = 100 * aspect;
    svg.setAttribute("viewBox", `0 0 ${W.toFixed(2)} 100`);
    if (!lm.length || mode === "minimal") return;

    const seen = (i) => lm[i] && lm[i][2] >= thresh;
    const x = (i) => lm[i][0] * W;
    const y = (i) => lm[i][1] * 100;
    const pt = (i) => [x(i), y(i)];

    const metrics = {};
    for (const m of (posture && posture.metrics) || []) metrics[m.key] = m;

    // One accent for the whole figure, from the worst axis — the same rule
    // the score uses, so the picture and the number agree about severity.
    // A metric the model does not trust is left out of it entirely.
    const rows = Object.values(metrics).filter((m) => !m.low_confidence);
    const worst = rows.reduce((acc, m) => Math.max(acc, m.ratio || 0), 0);
    const accent = rows.length ? colourFor(toneForRatio(worst)) : "var(--text-faint)";

    const mid = (a, b) => {
      if (seen(a) && seen(b)) return [(x(a) + x(b)) / 2, (y(a) + y(b)) / 2];
      if (seen(a)) return pt(a);
      if (seen(b)) return pt(b);
      return null;
    };

    const ctx = { svg, el, arcPath, lm, seen, pt, mid, W, accent, metrics, show,
                  baseline: opts.baseline, thresh };
    if (show.ghost) drawGhost(ctx);
    if (show.limbs) drawLimbs(ctx);
    if (cam.role === "front") drawFront(ctx); else drawSide(ctx);
    if (show.joints) drawJoints(ctx);
  }

  return { node: svg, render, modes: MODES };
}

function colourFor(tone) {
  return tone === "bad" ? "var(--bad)"
    : tone === "warn" ? "var(--warn)"
    : tone === "good" ? "var(--good)" : "var(--text-faint)";
}

/* ── shared pieces ─────────────────────────────────────────────────────── */

function line(ctx, [x1, y1], [x2, y2], attrs = {}) {
  ctx.svg.append(ctx.el("line", {
    x1, y1, x2, y2, "stroke-linecap": "round",
    stroke: attrs.stroke || ctx.accent,
    "stroke-width": attrs.width || 0.4,
    "stroke-dasharray": attrs.dash || null,
    opacity: attrs.opacity ?? 0.9,
  }));
}

/** A dashed axis through a point. The thing every angle is measured from, and
 *  the thing you are trying to line up with, so it is worth drawing. */
function reference(ctx, [px, py], horizontal, span = 26) {
  const a = horizontal ? [px - span, py] : [px, py - span];
  const b = horizontal ? [px + span, py] : [px, py + span * 0.8];
  line(ctx, a, b, { stroke: "var(--text-faint)", width: 0.22, dash: "1.1 1.6",
                    opacity: 0.55 });
}

/**
 * An angle arc at a vertex with its value beside it.
 *
 * The label is the server's number, not one recomputed from the pixels here.
 * Precision follows the reading: a tenth of a degree on a landmark that moves
 * by whole degrees between frames is decoration, so anything the model is less
 * than sure of is shown whole.
 */
function angle(ctx, vertex, fromAngle, toAngle, metric, radius = 9) {
  if (!metric || metric.value === null || metric.value === undefined) return;
  const [vx, vy] = vertex;
  const { d, mid } = ctx.arcPath(vx, vy, fromAngle, toAngle, radius);
  const low = metric.low_confidence;
  const stroke = low ? "var(--text-faint)" : ctx.accent;
  ctx.svg.append(ctx.el("path", {
    d, fill: "none", stroke, "stroke-width": 0.3, opacity: low ? 0.4 : 0.85,
  }));
  const lx = vx + (radius + 5.5) * Math.cos(mid);
  const ly = vy + (radius + 5.5) * Math.sin(mid);
  ctx.svg.append(ctx.el("text", {
    x: lx, y: ly, fill: stroke, "font-size": 3.4, "font-family": "var(--mono)",
    "text-anchor": "middle", "dominant-baseline": "middle",
    opacity: low ? 0.5 : 0.95,
  }, formatAngle(metric)));
}

/** Degrees with a unit, or the metric's own unit when it is not an angle. */
function formatAngle(metric) {
  const v = Number(metric.value);
  if (metric.unit !== "deg") return `${v.toFixed(2)}×`;
  // Whole degrees unless the model is confident: see the docstring above.
  const digits = (metric.confidence ?? 1) >= 0.9 ? 1 : 0;
  return `${v.toFixed(digits)}°`;
}

/* ── side view ─────────────────────────────────────────────────────────── */

/**
 * Ear over shoulder over hip: the chain every side metric is taken from.
 *
 * Only the near side is drawn. A side camera sees one ear and one shoulder;
 * drawing both would put a line through the far ones the model has inferred
 * rather than seen, which is exactly the geometry the app refuses to measure.
 */
function drawSide(ctx) {
  const { seen, pt, metrics, show } = ctx;
  const near = nearSide(ctx);
  if (!near) return;
  const [earI, shI, hipI] = near;
  const hasEar = seen(earI), hasSh = seen(shI), hasHip = seen(hipI);
  if (!hasSh) return;

  const shoulder = pt(shI);
  if (show.refs) {
    reference(ctx, shoulder, false);
    reference(ctx, shoulder, true, 14);
  }

  const UP = -Math.PI / 2;
  if (hasEar && show.geometry) {
    const ear = pt(earI);
    line(ctx, ear, shoulder, { width: 0.5 });
    if (show.arcs) {
      angle(ctx, shoulder, UP, Math.atan2(ear[1] - shoulder[1], ear[0] - shoulder[0]),
            metrics.neck_tilt, 10);
    }
    // Forward head is a horizontal offset, not an angle, so it gets a
    // horizontal connector rather than an arc: the picture should show what
    // the number is, not merely sit near it.
    if (show.arcs && metrics.forward_head) {
      line(ctx, [shoulder[0], ear[1]], ear,
           { width: 0.3, dash: "1.0 1.2", opacity: 0.8 });
      ctx.svg.append(ctx.el("text", {
        x: (shoulder[0] + ear[0]) / 2, y: ear[1] - 2.6,
        fill: ctx.accent, "font-size": 3.0, "font-family": "var(--mono)",
        "text-anchor": "middle", opacity: 0.85,
      }, formatAngle(metrics.forward_head)));
    }
  }

  if (hasHip && show.geometry) {
    const hip = pt(hipI);
    line(ctx, shoulder, hip, { width: 0.5, opacity: 0.85 });
    if (show.arcs) {
      angle(ctx, shoulder, UP + Math.PI,
            Math.atan2(hip[1] - shoulder[1], hip[0] - shoulder[0]),
            metrics.torso_lean, 11);
    }
  }
}

/** Which ear/shoulder/hip triple the camera can actually see. */
function nearSide(ctx) {
  const { seen } = ctx;
  const left = [P.leftEar, P.leftShoulder, P.leftHip];
  const right = [P.rightEar, P.rightShoulder, P.rightHip];
  const score = (c) => (seen(c[0]) ? 2 : 0) + (seen(c[1]) ? 2 : 0) + (seen(c[2]) ? 1 : 0);
  const best = score(right) >= score(left) ? right : left;
  return score(best) > 0 ? best : null;
}

/* ── front view ────────────────────────────────────────────────────────── */

/** Eye/ear line, shoulder line, and the true vertical between them. */
function drawFront(ctx) {
  const { seen, pt, mid, metrics, show } = ctx;
  const shoulders = seen(P.leftShoulder) && seen(P.rightShoulder);
  const ears = seen(P.leftEar) && seen(P.rightEar);
  const shMid = mid(P.leftShoulder, P.rightShoulder);
  if (!shMid) return;

  if (show.refs) reference(ctx, shMid, false, 30);

  if (shoulders && show.geometry) {
    const l = pt(P.leftShoulder), r = pt(P.rightShoulder);
    if (show.refs) {
      line(ctx, [Math.min(l[0], r[0]) - 6, shMid[1]], [Math.max(l[0], r[0]) + 6, shMid[1]],
           { stroke: "var(--text-faint)", width: 0.3, dash: "1.4 1.8", opacity: 0.5 });
    }
    line(ctx, l, r, { width: 0.5 });
    if (show.arcs) {
      const left = l[0] < r[0] ? l : r;
      const right = l[0] < r[0] ? r : l;
      angle(ctx, left, 0, Math.atan2(right[1] - left[1], right[0] - left[0]),
            metrics.shoulder_tilt, 12);
    }
  }

  if (ears && show.geometry) {
    const l = pt(P.leftEar), r = pt(P.rightEar);
    line(ctx, l, r, { width: 0.38, opacity: 0.8 });
    if (show.arcs) {
      const left = l[0] < r[0] ? l : r;
      const right = l[0] < r[0] ? r : l;
      angle(ctx, left, 0, Math.atan2(right[1] - left[1], right[0] - left[0]),
            metrics.head_roll, 9);
    }
  }

  // Lateral offset is a sideways displacement of the head over the shoulders,
  // so it is drawn as the displacement: a connector from the true vertical to
  // where the head midpoint actually is.
  const earMid = mid(P.leftEar, P.rightEar);
  if (earMid && show.geometry) {
    line(ctx, shMid, earMid, { width: 0.38, opacity: 0.7 });
    if (show.arcs && metrics.lateral_offset
        && Math.abs(earMid[0] - shMid[0]) > 0.4) {
      line(ctx, [shMid[0], earMid[1]], earMid,
           { width: 0.3, dash: "1.0 1.2", opacity: 0.8 });
      ctx.svg.append(ctx.el("text", {
        x: (shMid[0] + earMid[0]) / 2, y: earMid[1] - 2.6,
        fill: ctx.accent, "font-size": 3.0, "font-family": "var(--mono)",
        "text-anchor": "middle", opacity: 0.85,
      }, formatAngle(metrics.lateral_offset)));
    }
  }
}

/* ── ghost, limbs, joints ──────────────────────────────────────────────── */

/**
 * Where you were sitting when you set the baseline.
 *
 * Faint and dashed so the live figure stays the subject, with an arrow only
 * where a landmark has actually moved — an arrow per point would be a hedgehog
 * and would say nothing. Points the calibration barely saw are left out: a
 * median of three frames is not a position.
 */
function drawGhost(ctx) {
  const base = ctx.baseline || [];
  if (!base.length) return;
  const { W } = ctx;
  const bx = (i) => base[i][0] * W;
  const by = (i) => base[i][1] * 100;
  const solid = (i) => base[i] && base[i][2] >= 0.5;

  const links = [
    [P.leftEar, P.leftShoulder], [P.rightEar, P.rightShoulder],
    [P.leftShoulder, P.rightShoulder], [P.leftShoulder, P.leftHip],
    [P.rightShoulder, P.rightHip], [P.leftHip, P.rightHip],
  ];
  for (const [a, b] of links) {
    if (!solid(a) || !solid(b)) continue;
    ctx.svg.append(ctx.el("line", {
      x1: bx(a), y1: by(a), x2: bx(b), y2: by(b),
      stroke: "var(--accent)", "stroke-width": 0.3,
      "stroke-dasharray": "1.4 1.6", opacity: 0.45,
    }));
  }
  for (let i = 0; i < base.length; i++) {
    if (!solid(i)) continue;
    ctx.svg.append(ctx.el("circle", {
      cx: bx(i), cy: by(i), r: 0.6, fill: "none",
      stroke: "var(--accent)", "stroke-width": 0.24, opacity: 0.55,
    }));
    if (!ctx.seen(i)) continue;
    const [cx, cy] = ctx.pt(i);
    const dx = cx - bx(i), dy = cy - by(i);
    if (Math.hypot(dx, dy) < 2.2) continue;   // moved less than it wobbles
    const len = Math.hypot(dx, dy);
    const back = 1.4 / len;
    ctx.svg.append(ctx.el("line", {
      x1: bx(i), y1: by(i),
      x2: cx - dx * back, y2: cy - dy * back,
      stroke: "var(--accent)", "stroke-width": 0.28, opacity: 0.6,
      "marker-end": null,
    }));
  }
}

/** Faint limbs, so the figure reads as a body rather than a diagram. */
function drawLimbs(ctx) {
  const links = [
    [P.leftShoulder, P.leftElbow], [P.rightShoulder, P.rightElbow],
    [P.leftHip, P.leftKnee], [P.rightHip, P.rightKnee],
    [P.leftEar, P.leftEye], [P.rightEar, P.rightEye],
    [P.leftEye, P.nose], [P.rightEye, P.nose],
  ];
  for (const [a, b] of links) {
    if (!ctx.seen(a) || !ctx.seen(b)) continue;
    line(ctx, ctx.pt(a), ctx.pt(b),
         { stroke: "var(--text-faint)", width: 0.25, opacity: 0.65 });
  }
}

/**
 * The tracked points themselves.
 *
 * Landmarks below the threshold are drawn, in the bad colour and hollow. That
 * is deliberate: "not tracked" and "tracked but not trusted" are different
 * states, and the second is the one you can fix by moving.
 */
function drawJoints(ctx) {
  const { lm, thresh, show } = ctx;
  const primary = new Set([P.leftEar, P.rightEar, P.leftShoulder, P.rightShoulder,
                           P.leftHip, P.rightHip]);
  for (let i = 0; i < lm.length; i++) {
    const conf = lm[i][2];
    if (conf < 0.05) continue;
    if (!show.secondary && !primary.has(i)) continue;
    const faint = conf < thresh;
    const [cx, cy] = ctx.pt(i);
    ctx.svg.append(ctx.el("circle", {
      cx, cy, r: primary.has(i) ? 0.75 : 0.45,
      fill: faint ? "none" : (primary.has(i) ? ctx.accent : "var(--text-dim)"),
      stroke: faint ? "var(--bad)" : "none",
      "stroke-width": faint ? 0.25 : 0,
      opacity: faint ? 0.6 : (primary.has(i) ? 0.95 : 0.6),
    }));
  }
}
