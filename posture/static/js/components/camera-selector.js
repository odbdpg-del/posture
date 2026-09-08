import { h, mount, reconcile } from "../dom.js";

const TONE = {
  tracking: "ok", partial: "ok", cannot_see: "warn",
  no_person: "", error: "bad", stopped: "", starting: "",
};

/**
 * Compact chips for switching the primary feed.
 *
 * Secondary cameras are selectors, not equally-sized diagnostic cards: one
 * feed is the subject and the rest are one click away.
 */
export function CameraSelector(store) {
  const root = h("div", { class: "thumbs" });

  function render() {
    const cams = (store.status && store.status.cameras) || [];
    const primary = store.primary();
    if (cams.length < 2) { mount(root); root.dataset.sig = ""; return; }

    reconcile(root, cams.map((c) => `${c.index}:${c.name}:${c.role}`).join("|"),
      () => cams.map((c) => h("button", {
        class: "thumb", dataset: { index: String(c.index) },
        onClick: () => store.set({ primaryCamera: c.index }),
      }, h("span", { class: "led" }), c.name, h("span", { class: "faint" }, c.role))));

    for (const c of cams) {
      const btn = root.querySelector(`[data-index="${c.index}"]`);
      if (!btn) continue;
      btn.className = `thumb ${TONE[c.state] || ""}` +
        (primary && c.index === primary.index ? " active" : "");
      btn.title = `${c.name} — ${c.state_label}`;
    }
  }

  render();
  store.subscribe(render);
  return root;
}
