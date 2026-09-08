import { h, mount, shortDuration, reconcile } from "../dom.js";
import { api } from "../api.js";

/**
 * One day's totals, straight from SQLite.
 *
 * The four figures the brief asks for — good versus bad time, the longest
 * streak, and how many alerts fired — plus a bar per bucket across the day.
 *
 * Time the app was closed is attributed to neither good nor bad; the bar for
 * that stretch is simply absent rather than filled in with a guess.
 */
export function DayStats(store) {
  const picker = h("select", { class: "headsel", onChange: (e) => load(e.target.value) });
  const figures = h("div", { class: "figures" });
  const chart = h("div", { class: "daychart" });
  const note = h("div", { class: "faint" });

  const root = h("section", { class: "panel" },
    h("div", { class: "panel-head" },
      h("span", null, "Daily summary"),
      h("span", { style: { flex: "1" } }),
      picker),
    h("div", { class: "panel-body" }, figures, chart, note));

  let day = null;
  let days = [];

  async function loadDays() {
    try {
      const res = await api.statsDays();
      days = res.days || [];
      reconcile(picker, days.join(","), () =>
        (days.length ? days : ["today"]).map((d) =>
          h("option", { value: d }, d)));
      if (!day) day = days[0] || null;
      if (day) picker.value = day;
    } catch { /* the panel simply shows nothing */ }
  }

  async function load(which) {
    day = which || day;
    try {
      render(await api.stats(day));
    } catch (err) {
      mount(figures);
      note.textContent = String(err.message || err);
    }
  }

  function figure(label, value, sub) {
    return h("div", { class: "figure" },
      h("div", { class: "flabel" }, label),
      h("div", { class: "fvalue num" }, value),
      sub ? h("div", { class: "fsub" }, sub) : null);
  }

  function render(stats) {
    if (!stats.available) {
      mount(figures);
      mount(chart);
      note.textContent = stats.reason || "Stats are unavailable.";
      return;
    }
    note.textContent = "";
    const pct = stats.good_percent;
    mount(figures,
      figure("Good posture", pct === null ? "—" : `${pct}%`,
        `${shortDuration(stats.good_seconds)} of ${shortDuration(stats.measured_seconds)} measured`),
      figure("Longest streak", shortDuration(stats.longest_good_streak), "unbroken"),
      figure("Alerts", String(stats.alerts),
        `${stats.episodes} bad stretch${stats.episodes === 1 ? "" : "es"}`),
      figure("Average score", stats.average_score === null ? "—"
        : String(stats.average_score), "across the day"),
    );

    const buckets = stats.timeline || [];
    if (!buckets.length) {
      mount(chart, h("div", { class: "muted", style: { padding: "18px 0" } },
        "Nothing recorded for this day yet."));
      return;
    }
    const max = Math.max(...buckets.map((b) => b.good + b.bad + b.other), 1);
    mount(chart,
      h("div", { class: "daybars" }, buckets.map((b) => {
        const total = b.good + b.bad + b.other;
        const height = total ? Math.max(6, Math.round((total / max) * 100)) : 0;
        const goodPct = total ? (b.good / total) * 100 : 0;
        const badPct = total ? (b.bad / total) * 100 : 0;
        const when = new Date(b.t * 1000).toLocaleTimeString([],
          { hour: "numeric", minute: "2-digit" });
        return h("span", {
          class: `daybar ${total ? "" : "empty"}`, style: { height: `${height}%` },
          title: total
            ? `${when} — ${Math.round(goodPct)}% good, ${Math.round(badPct)}% bad`
              + (b.score === null ? "" : `, score ${b.score}`)
            : `${when} — not at the desk`,
        },
          h("i", { class: "g", style: { height: `${goodPct}%` } }),
          h("i", { class: "b", style: { height: `${badPct}%` } }));
      })),
      h("div", { class: "dayaxis" }, [0, 6, 12, 18, 24].map((hour) =>
        h("span", null, `${String(hour % 24).padStart(2, "0")}:00`))));
  }

  loadDays().then(() => load(day));
  // Refreshed on a slow timer: a daily total does not move quickly, and the
  // live view already covers the last half hour.
  setInterval(() => { loadDays(); load(day); }, 30000);
  return root;
}
