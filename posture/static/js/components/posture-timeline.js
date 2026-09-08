import { h, mount, clockTime, shortDuration } from "../dom.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const W = 1000;   // viewBox units; the SVG scales to whatever width it gets
const HEIGHT = 120;
const PAD_L = 30;
const PAD_R = 8;
const PAD_T = 8;
const PAD_B = 18;

/**
 * Score over time, with the bad stretches marked.
 *
 * Fed from the server's in-memory history, which records the scores the app
 * actually computed. When there is no data yet it says so rather than drawing
 * a plausible-looking line — an invented trend is worse than an empty chart.
 */
export function PostureTimeline(store, { compact = false } = {}) {
  const chart = h("div", { class: "timeline" });
  const tip = h("div", { class: "tt", hidden: true });
  const root = h("section", { class: "panel" },
    h("div", { class: "panel-head" },
      h("span", null, "Posture history"),
      h("span", { class: "spacer", style: { flex: "1" } }),
      h("span", { class: "faint" }, "last 30 minutes")),
    h("div", { class: "panel-body" }, chart, tip));

  function el(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  let signature = null;

  function render() {
    const hist = store.history;
    const samples = (hist && hist.samples) || [];
    // The chart only changes when the history poll brings new data (every few
    // seconds), not on every status tick. Redrawing it at 2 Hz replaced the
    // hovered episode band, so its mouseleave never fired and the tooltip was
    // left stranded on screen.
    const sig = `${samples.length}:${samples.length ? samples[samples.length - 1].t : 0}` +
      `:${(hist && hist.episodes || []).length}` +
      `:${(hist && hist.episodes || []).map((e) => e.ended || "x").join(",")}`;
    if (sig === signature) return;
    signature = sig;
    hideTip();
    const scored = samples.filter((s) => s.score !== null && s.score !== undefined);

    if (scored.length < 2) {
      mount(chart, h("div", { class: "empty" },
        hist && samples.length
          ? "Collecting — scores appear once posture can be measured."
          : "No history yet. It fills as you work."));
      return;
    }

    const t0 = samples[0].t;
    const t1 = samples[samples.length - 1].t;
    const span = Math.max(1, t1 - t0);
    const height = compact ? 90 : HEIGHT;
    const x = (t) => PAD_L + ((t - t0) / span) * (W - PAD_L - PAD_R);
    const y = (v) => PAD_T + (1 - v / 100) * (height - PAD_T - PAD_B);

    const svg = el("svg", {
      viewBox: `0 0 ${W} ${height}`, preserveAspectRatio: "none",
      style: `height:${height}px`,
    });

    for (const level of [100, 80, 60, 40, 20]) {
      svg.append(el("line", {
        x1: PAD_L, y1: y(level), x2: W - PAD_R, y2: y(level),
        stroke: "var(--border-soft)", "stroke-width": 1,
      }));
      svg.append(el("text", {
        x: PAD_L - 6, y: y(level) + 3, "text-anchor": "end",
        fill: "var(--text-faint)", "font-size": 9,
      }, String(level)));
    }

    // Bad episodes as bands behind the trace, so a glance shows when rather
    // than just how much.
    for (const ep of (hist.episodes || [])) {
      const from = Math.max(ep.started, t0);
      const to = Math.min(ep.ended || t1, t1);
      if (to <= from) continue;
      const band = el("rect", {
        x: x(from), y: PAD_T, width: Math.max(1.5, x(to) - x(from)),
        height: height - PAD_T - PAD_B,
        fill: "var(--bad)", opacity: 0.12, style: "cursor:pointer",
      });
      band.addEventListener("mousemove", (e) => showTip(e, episodeTip(ep)));
      band.addEventListener("mouseleave", hideTip);
      svg.append(band);
    }

    // Gaps (away, unreadable) break the line rather than being interpolated
    // across, which would imply measurements that never happened.
    let d = "";
    let pen = false;
    for (const s of samples) {
      if (s.score === null || s.score === undefined) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${x(s.t).toFixed(1)},${y(s.score).toFixed(1)} `;
      pen = true;
    }
    svg.append(el("path", {
      d, fill: "none", stroke: "var(--good)", "stroke-width": 1.6,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));

    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const t = t0 + (span * i) / ticks;
      svg.append(el("text", {
        x: x(t), y: height - 5,
        "text-anchor": i === 0 ? "start" : i === ticks ? "end" : "middle",
        fill: "var(--text-faint)", "font-size": 9,
      }, clockTime(t)));
    }

    mount(chart, svg);
  }

  function episodeTip(ep) {
    const rows = [
      h("div", null, clockTime(ep.started)),
      h("div", null, h("span", { class: "k" }, "Duration "),
        shortDuration(ep.duration)),
    ];
    for (const [key, value] of Object.entries(ep.peak || {})) {
      rows.push(h("div", null,
        h("span", { class: "k" }, `${key.replace(/_/g, " ")} `),
        `peak +${value}`));
    }
    if (ep.ongoing) rows.push(h("div", { class: "k" }, "ongoing"));
    return rows;
  }

  function showTip(event, nodes) {
    const box = root.getBoundingClientRect();
    mount(tip, nodes);
    tip.hidden = false;
    tip.style.left = `${event.clientX - box.left + 12}px`;
    tip.style.top = `${event.clientY - box.top - 8}px`;
  }
  function hideTip() { tip.hidden = true; }

  // Belt and braces: leaving the panel altogether, or scrolling it away,
  // should also clear a tooltip that is somehow still showing.
  root.addEventListener("mouseleave", hideTip);
  chart.addEventListener("mouseleave", hideTip);
  window.addEventListener("blur", hideTip);

  render();
  store.subscribe(render);
  return root;
}
