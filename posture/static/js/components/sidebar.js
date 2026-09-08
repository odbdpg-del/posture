import { h } from "../dom.js";

export const SECTIONS = [
  { id: "live", label: "Live", icon: "◉" },
  { id: "history", label: "History", icon: "▤" },
  { id: "cameras", label: "Cameras", icon: "▢" },
  { id: "calibration", label: "Calibration", icon: "⌖" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

export function Sidebar(store, onNavigate) {
  const root = h("nav", { class: "sidebar" });

  // Built once. These are the app's primary controls and must never be
  // replaced under the pointer — see reconcile() in dom.js for why.
  const badge = h("span", { class: "navbadge", hidden: true });
  const items = SECTIONS.map((s) => h("button", {
    class: "navitem", title: s.label,
    onClick: () => onNavigate(s.id),
  },
    h("span", { class: "ic" }, s.icon),
    h("span", { class: "lbl" }, s.label),
    s.id === "live" ? badge : null,
  ));
  const collapseIcon = h("span", { class: "ic" }, "«");
  const collapse = h("button", {
    class: "navitem", title: "Collapse navigation",
    onClick: () => store.set({ sidebarCollapsed: !store.sidebarCollapsed }),
  }, collapseIcon, h("span", { class: "lbl" }, "Collapse"));

  root.append(...items, h("span", { style: { flex: "1" } }), collapse);

  function render() {
    SECTIONS.forEach((s, i) => {
      items[i].classList.toggle("active", store.view === s.id);
    });
    const alerts = activeAlertCount(store);
    badge.hidden = alerts === 0;
    badge.textContent = String(alerts);
    collapseIcon.textContent = store.sidebarCollapsed ? "»" : "«";
  }

  render();
  store.subscribe(render);
  return root;
}

function activeAlertCount(store) {
  const p = store.status && store.status.posture;
  return p && p.state === "bad" ? (p.offenders || []).length : 0;
}
