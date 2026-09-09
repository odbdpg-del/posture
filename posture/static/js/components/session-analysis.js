import { h, mount, reconcile, shortDuration } from "../dom.js";

/**
 * One chart per metric, plus what the session adds up to.
 *
 * The timeline shows the score, which is the worst axis at each instant. It
 * cannot show forward head creeping up all morning while everything else
 * holds — the score would just say "fine" until the moment it did not. A
 * track per metric can, and that is the shape of the problem people actually
 * have.
 *
 * The baseline is a dashed rule across each chart and the tolerance band is
 * shaded, so "how far past normal" is read off the picture rather than
 * arithmetic. Deviation is drawn as a filled area only where it is past
 * tolerance: the eye should go to the sustained stretches, which are the ones
 * worth acting on, not to every excursion.
 *
 * Charts are SVG built from the same points the server sends. Nothing is
 * interpolated across a gap where nobody was measured — a line drawn through
 * the time you were at lunch would be an invented posture.
 */
export function SessionAnalysis(store) {
  const body = h("div", { class: "panel-body session-body" });
  const summary = h("div", { class: "biomech-summary" });
  const root = h("section", { class: "panel" },
    h("div", { class: "panel-head" },
      h("span", null, "Session analysis"),
      h("span", { style: { flex: "1" } }), summary),
    body);

  const charts = new Map();

  function render() {
    const session = (store.history && store.history.session) || null;
    if (!session) return;
    const tracks = (session.tracks || []).filter((t) => t.points.length > 1);

    if (!tracks.length) {
      reconcile(body, "empty", () => h("div", { class: "muted" },
        "Charts appear once there are a couple of minutes of measurements. "
        + "Nothing is drawn before then rather than a flat line at zero."));
      mount(summary);
      return;
    }

    reconcile(body, tracks.map((t) => t.key).join(","), () => {
      charts.clear();
      return tracks.map((t) => {
        const chart = MetricChart(t);
        charts.set(t.key, chart);
        return chart.node;
      });
    });
    for (const t of tracks) {
      const chart = charts.get(t.key);
      if (chart) chart.update(t, session);
    }

    const v = session.variability || {};
    mount(summary,
      h("span", { class: "faint" }, `Session ${shortDuration(session.measured)}`),
      v.band ? h("span", { class: "tag" }, `${v.band} movement`) : null,
      v.changes_per_hour !== null && v.changes_per_hour !== undefined
        ? h("span", { class: "faint" }, `${v.changes_per_hour}/h position changes`)
        : null);
  }

  render();
  store.subscribe(render);
  return root;
}

// The plot stretches to the panel width, so nothing inside it may be text:
// preserveAspectRatio="none" scales glyphs with the box and a chart twice as
// wide as its viewBox renders every label at twice the letter width. Geometry
// stretches harmlessly; type does not. So the SVG holds the plot and every
// label is HTML beside it.
const W = 600, H = 88, PAD_L = 2, PAD_R = 2, PAD_T = 8, PAD_B = 8;
const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs, text) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, v);
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function MetricChart(track) {
  const svg = el("svg", {
    class: "chart", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none",
  });
  const stats = h("div", { class: "chart-stats" });
  const baseTag = h("span", { class: "chart-base" });
  const axisFrom = h("span", null);
  const axisTo = h("span", null);
  const node = h("div", { class: "chart-block" },
    h("div", { class: "chart-head" },
      h("span", { class: "chart-title" }, track.label), baseTag, stats),
    h("div", { class: "chart-wrap" }, svg),
    h("div", { class: "chart-axis" }, axisFrom, axisTo));

  function update(t, session) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const pts = t.points || [];
    if (pts.length < 2) return;

    const base = t.baseline;
    const tol = t.tolerance || 0;
    const values = pts.map((p) => p.v);
    // The band always fits, so a metric sitting quietly inside tolerance
    // renders as a flat line inside a visible band rather than as noise
    // amplified to fill the chart.
    const lo = Math.min(...values, base !== null ? base - tol * 1.3 : Infinity);
    const hi = Math.max(...values, base !== null ? base + tol * 1.3 : -Infinity);
    const span = (hi - lo) || 1;
    const t0 = pts[0].t, t1 = pts[pts.length - 1].t;
    const dt = (t1 - t0) || 1;

    const px = (p) => PAD_L + ((p.t - t0) / dt) * (W - PAD_L - PAD_R);
    const py = (v) => PAD_T + (1 - (v - lo) / span) * (H - PAD_T - PAD_B);

    if (base !== null && base !== undefined) {
      if (tol > 0) {
        const top = py(base + tol), bottom = py(base - tol);
        svg.append(el("rect", {
          x: PAD_L, y: Math.min(top, bottom), width: W - PAD_L - PAD_R,
          height: Math.abs(bottom - top), fill: "var(--good)", opacity: 0.055,
        }));
      }
      svg.append(el("line", {
        x1: PAD_L, y1: py(base), x2: W - PAD_R, y2: py(base),
        stroke: "var(--text-faint)", "stroke-width": 1,
        "stroke-dasharray": "4 4", opacity: 0.7,
      }));
    }
    baseTag.textContent = (base === null || base === undefined)
      ? "" : `baseline ${fmtShort(base, t.unit)}`;

    // Split at gaps so nothing is drawn across time nobody measured.
    const gap = (session.interval || 5) * 4;
    let run = [];
    const runs = [];
    for (const p of pts) {
      if (run.length && p.t - run[run.length - 1].t > gap) { runs.push(run); run = []; }
      run.push(p);
    }
    if (run.length) runs.push(run);

    for (const segment of runs) {
      if (segment.length < 2) continue;
      const d = segment.map((p, i) => `${i ? "L" : "M"} ${px(p).toFixed(1)} ${py(p.v).toFixed(1)}`)
        .join(" ");
      svg.append(el("path", {
        d, fill: "none", stroke: "var(--text-dim)", "stroke-width": 1.2,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      }));
      // Past tolerance, in the accent, over the top of the neutral line.
      let over = [];
      for (const p of segment.concat([null])) {
        if (p && p.over) { over.push(p); continue; }
        if (over.length > 1) {
          svg.append(el("path", {
            d: over.map((q, i) => `${i ? "L" : "M"} ${px(q).toFixed(1)} ${py(q.v).toFixed(1)}`)
              .join(" "),
            fill: "none", stroke: "var(--warn)", "stroke-width": 1.6,
            "stroke-linejoin": "round", "stroke-linecap": "round",
          }));
        }
        over = [];
      }
    }

    axisFrom.textContent = clock(t0);
    axisTo.textContent = clock(t1);

    mount(stats,
      t.total_over > 0
        ? h("span", null, `${shortDuration(t.total_over)} past tolerance`)
        : h("span", { class: "faint" }, "within tolerance throughout"),
      t.total_over > 0
        ? h("span", { class: "faint" }, `${Math.round(t.share_over * 100)}% of session`)
        : null,
      t.longest_over >= (session.min_run || 5)
        ? h("span", { class: "faint" }, `longest ${shortDuration(t.longest_over)}`)
        : null);
  }

  return { node, update };
}

function clock(epoch) {
  return new Date(epoch * 1000).toLocaleTimeString([],
    { hour: "numeric", minute: "2-digit" });
}

function fmtShort(v, unit) {
  const suffix = unit === "deg" ? "°" : "";
  return `${Number(v).toFixed(Math.abs(v) < 1 ? 2 : 0)}${suffix}`;
}
