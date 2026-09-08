import { toneForRatio } from "../dom.js";

// Landmark order as sent by the server (monitor.PREVIEW_LANDMARKS).
const P = {
  nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6, leftElbow: 7, rightElbow: 8,
  leftHip: 9, rightHip: 10, leftKnee: 11, rightKnee: 12,
};

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Posture geometry drawn over the live frame.
 *
 * Deliberately sparse. The frame already shows a person; the overlay's job is
 * to name the two or three lines the app is actually measuring, not to redraw
 * the skeleton on top of the skeleton. Colour follows posture state so a
 * glance at the feed says as much as the gauges do.
 *
 * The frame arrives clean and everything here is vector, so the overlay can be
 * styled and coloured by posture state rather than baked into pixels at encode
 * time. Lines are thin and joints small: the video is the subject.
 */
export function PostureOverlay() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("viewBox", "0 0 100 100");

  function el(name, attrs) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  function colourFor(tone) {
    return tone === "bad" ? "var(--bad)"
      : tone === "warn" ? "var(--warn)"
      : tone === "good" ? "var(--good)" : "var(--text-faint)";
  }

  /**
   * @param cam     camera snapshot (landmarks, visibility)
   * @param posture verdict, used only for colour
   * @param thresh  visibility threshold, so we hide what the app itself ignores
   */
  function render(cam, posture, thresh, { skeletonOnly = false } = {}) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const lm = (cam && cam.landmarks) || [];
    if (!lm.length) return;

    const seen = (i) => lm[i] && lm[i][2] >= thresh;
    const x = (i) => lm[i][0] * 100;
    const y = (i) => lm[i][1] * 100;

    // Worst current ratio drives the overlay colour, matching the score's
    // "as bad as your worst axis" rule.
    const metrics = (posture && posture.metrics) || [];
    const worst = metrics.reduce((acc, m) => Math.max(acc, m.ratio || 0), 0);
    const stroke = colourFor(metrics.length ? toneForRatio(worst) : "idle");

    const shouldersVisible = seen(P.leftShoulder) && seen(P.rightShoulder);
    const midShoulder = shouldersVisible
      ? [(x(P.leftShoulder) + x(P.rightShoulder)) / 2,
         (y(P.leftShoulder) + y(P.rightShoulder)) / 2]
      : seen(P.leftShoulder) ? [x(P.leftShoulder), y(P.leftShoulder)]
      : seen(P.rightShoulder) ? [x(P.rightShoulder), y(P.rightShoulder)] : null;

    // Vertical reference through the shoulders: the line every angle in the
    // app is measured from, so it is worth showing explicitly.
    if (midShoulder) {
      svg.append(el("line", {
        x1: midShoulder[0], y1: Math.max(0, midShoulder[1] - 34),
        x2: midShoulder[0], y2: Math.min(100, midShoulder[1] + 26),
        stroke: "var(--text-faint)", "stroke-width": 0.35,
        "stroke-dasharray": "1.6 1.6", opacity: 0.55,
      }));
    }

    if (shouldersVisible) {
      svg.append(el("line", {
        x1: x(P.leftShoulder), y1: y(P.leftShoulder),
        x2: x(P.rightShoulder), y2: y(P.rightShoulder),
        stroke, "stroke-width": 0.55, "stroke-linecap": "round", opacity: 0.85,
      }));
    }

    // Head centreline: ear midpoint down to the shoulders. This is the segment
    // neck tilt measures, which is the metric that survives without hips.
    const ears = [P.leftEar, P.rightEar].filter(seen);
    if (ears.length && midShoulder) {
      const ex = ears.reduce((a, i) => a + x(i), 0) / ears.length;
      const ey = ears.reduce((a, i) => a + y(i), 0) / ears.length;
      svg.append(el("line", {
        x1: ex, y1: ey, x2: midShoulder[0], y2: midShoulder[1],
        stroke, "stroke-width": 0.55, "stroke-linecap": "round", opacity: 0.85,
      }));
      svg.append(el("circle", { cx: ex, cy: ey, r: 0.7, fill: stroke }));
    }

    // Torso centreline, only when hips are actually in frame. At a desk they
    // usually are not, and drawing a line to a landmark the app is ignoring
    // would imply a measurement that is not happening.
    const hips = [P.leftHip, P.rightHip].filter(seen);
    if (hips.length && midShoulder) {
      const hx = hips.reduce((a, i) => a + x(i), 0) / hips.length;
      const hy = hips.reduce((a, i) => a + y(i), 0) / hips.length;
      svg.append(el("line", {
        x1: midShoulder[0], y1: midShoulder[1], x2: hx, y2: hy,
        stroke, "stroke-width": 0.55, "stroke-linecap": "round", opacity: 0.7,
      }));
      svg.append(el("circle", { cx: hx, cy: hy, r: 0.6, fill: stroke, opacity: 0.8 }));
    }

    if (midShoulder) {
      svg.append(el("circle", {
        cx: midShoulder[0], cy: midShoulder[1], r: 0.8, fill: stroke,
      }));
    }

    // With no picture underneath, the figure has to carry the framing on its
    // own, so the limbs are drawn too. Over video they would be redundant
    // clutter on top of a person who is already visible.
    if (skeletonOnly) {
      const links = [
        [P.leftShoulder, P.leftElbow], [P.rightShoulder, P.rightElbow],
        [P.leftHip, P.leftKnee], [P.rightHip, P.rightKnee],
        [P.leftEar, P.leftEye], [P.rightEar, P.rightEye],
        [P.leftEye, P.nose], [P.rightEye, P.nose],
      ];
      for (const [a, b] of links) {
        if (!seen(a) || !seen(b)) continue;
        svg.append(el("line", {
          x1: x(a), y1: y(a), x2: x(b), y2: y(b),
          stroke: "var(--text-faint)", "stroke-width": 0.4,
          "stroke-linecap": "round", opacity: 0.75,
        }));
      }
    }

    // The tracked joints themselves. They say "this is what I can see" -- a
    // little more present in skeleton mode, where nothing else is.
    for (let i = 0; i < lm.length; i++) {
      const conf = lm[i][2];
      if (conf < 0.05) continue;
      const faint = conf < thresh;
      svg.append(el("circle", {
        cx: x(i), cy: y(i),
        r: (faint ? 0.35 : 0.5) * (skeletonOnly ? 1.4 : 1),
        fill: faint ? "var(--bad)" : "var(--text)",
        opacity: faint ? 0.5 : (skeletonOnly ? 0.85 : 0.55),
      }));
    }
  }

  return { node: svg, render };
}
