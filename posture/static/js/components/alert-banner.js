import { h, mount, shortDuration } from "../dom.js";
import { api } from "../api.js";

const LEVEL_TEXT = {
  1: "Subtle", 2: "Notified", 3: "Full screen",
};

/**
 * The escalation, as the panel sees it.
 *
 * The alert itself is delivered by the OS notification and the full-screen
 * overlay; this is the readout — which step you are on, how long you have been
 * there, and the only two ways out. It never flashes: the thing it reports is
 * already at least a minute old by the time the detector calls it bad.
 */
export function AlertBanner(store) {
  const body = h("div", { class: "panel-body" });
  const root = h("section", { class: "panel" },
    h("div", { class: "panel-head" }, "Alert"), body);

  let signature = null;

  async function snooze() {
    await api.snooze();
  }
  async function resume() {
    await api.cancelSnooze();
  }

  function render() {
    const alert = (store.status && store.status.alert) || {};
    const enabled = store.config ? store.config.alerts.enabled : true;
    const sig = `${alert.level}:${alert.snoozed}:${enabled}`;
    const hold = alert.hold_remaining;

    if (sig !== signature) {
      signature = sig;
      if (!enabled) {
        mount(body, h("div", { class: "muted" },
          "Alerts are switched off under Settings."));
        return;
      }
      if (alert.snoozed) {
        mount(body,
          h("div", { class: "alertline snoozed" },
            h("span", { class: "lvl" }, "Snoozed"),
            h("span", { class: "rem faint" })),
          h("div", { class: "row", style: { marginTop: "10px" } },
            h("button", { class: "sm", onClick: resume }, "Resume alerts")));
        return;
      }
      if (!alert.active) {
        mount(body, h("div", { class: "alert-ok" }, "✓ Not alerting"));
        return;
      }
      mount(body,
        h("div", { class: `alertline lv${alert.level}` },
          h("span", { class: "lvl" }, LEVEL_TEXT[alert.level] || "Alert"),
          h("span", { class: "rem faint" })),
        h("div", { class: "headline" }, alert.headline || ""),
        alert.level >= 3
          ? h("div", { class: "holdbar" }, h("i")) : null,
        h("div", { class: "row", style: { marginTop: "10px" } },
          h("button", { class: "sm", onClick: snooze }, "Snooze")));
    }

    // Volatile bits update in place — see the rendering rule in dom.js.
    const rem = body.querySelector(".rem");
    if (rem) {
      rem.textContent = alert.snoozed
        ? `${shortDuration(alert.snooze_remaining)} left`
        : `for ${shortDuration(alert.in_level_for)}`;
    }
    const bar = body.querySelector(".holdbar i");
    if (bar && alert.hold_required) {
      const done = Math.max(0, Math.min(1, 1 - (hold || 0) / alert.hold_required));
      bar.style.width = `${Math.round(done * 100)}%`;
    }
  }

  render();
  store.subscribe(render);
  return root;
}
