import { h, mount, reconcile } from "../dom.js";
import { api } from "../api.js";
import { refreshConfig, refreshDevices } from "../store.js";

/**
 * Camera management: what is attached, whether it works, and what role it plays.
 *
 * Edits are staged on the store and applied by an Apply button that sits in
 * this section — the earlier version put the only Apply under Settings, so
 * ticking a checkbox here appeared to do nothing at all.
 */
export function CameraManager(store) {
  const banner = h("div");
  const list = h("div", { class: "list" });
  const note = h("span", { class: "faint" });
  const dirty = h("span", { class: "faint" });

  const applyBtn = h("button", { class: "primary", onClick: apply },
    "Apply camera changes");

  const root = h("div", { class: "panel" },
    h("div", { class: "panel-head" }, "Cameras"),
    h("div", { class: "panel-body" },
      banner, list,
      h("div", { class: "row", style: { marginTop: "12px" } },
        applyBtn,
        h("button", { onClick: () => rescan(false) }, "Rescan devices"),
        h("button", { onClick: () => rescan(true) }, "Test each camera"),
        dirty, note)));

  const thumbAt = new Map();

  async function rescan(check) {
    note.textContent = check ? "opening each camera…" : "scanning…";
    try {
      await refreshDevices(check);
      const d = store.devices;
      note.textContent = `${(d.devices || []).length} device(s) via ${d.method}`;
    } catch (err) { note.textContent = String(err.message || err); }
  }

  function staged() {
    if (!store.pendingCameras) {
      store.pendingCameras = JSON.parse(JSON.stringify(store.cameras()));
    }
    return store.pendingCameras;
  }

  function update(index, name, patch) {
    const cams = staged();
    let cam = cams.find((c) => c.index === index);
    if (!cam) {
      cam = { index, role: "side", name, enabled: false,
              width: 640, height: 480, backend: null };
      cams.push(cam);
    }
    Object.assign(cam, patch, { name });
    store.set({ pendingCameras: cams });
  }

  async function apply() {
    const cfg = JSON.parse(JSON.stringify(store.config));
    cfg.cameras = staged();
    applyBtn.disabled = true;
    try {
      const res = await api.saveConfig(cfg);
      if (res.ok) {
        store.set({ pendingCameras: null });
        await refreshConfig();
        show("ok", "Saved. Cameras restarting…");
      } else {
        show("bad", "Not saved", res.problems || []);
      }
    } catch (err) {
      show("bad", String(err.message || err));
    } finally { applyBtn.disabled = false; }
  }

  function show(kind, text, items = []) {
    mount(banner, h("div", { class: `banner ${kind}` }, text,
      items.length ? h("ul", null, items.map((p) => h("li", null, p))) : null));
    if (kind === "ok") setTimeout(() => mount(banner), 4000);
  }

  function render() {
    const devices = (store.devices && store.devices.devices) || [];
    const configured = staged();
    const running = (store.status && store.status.cameras) || [];
    // Frames only exist in video mode; asking for one in skeleton or off mode
    // just 404s. The retired boolean flag is no longer serialised, so testing
    // it here read undefined and silently meant "always on".
    const previewOn = !!(store.config && store.config.web.preview_mode === "video");
    dirty.textContent = store.pendingCameras
      && JSON.stringify(store.pendingCameras) !== JSON.stringify(store.cameras())
      ? "unsaved changes" : "";

    if (store.devices && store.devices.warning) {
      mount(banner, h("div", { class: "banner bad" }, store.devices.warning));
    }

    if (!devices.length) {
      mount(list, h("div", { class: "muted" }, "No cameras found. Try Rescan devices."));
      list.dataset.sig = "";
      return;
    }

    // Rows carry a checkbox and a dropdown, so they are built from the device
    // list and then only updated. Rebuilding them twice a second swallowed
    // clicks and would reset an open dropdown mid-choice.
    reconcile(list, devices.map((d) =>
      `${d.index}:${d.name}:${d.available}:${d.detail}`).join("|"), () =>
      devices.map((dev) => h("div", { class: "devrow", dataset: { index: String(dev.index) } },
        h("input", {
          type: "checkbox",
          onChange: (e) => update(dev.index, dev.name, { enabled: e.target.checked }),
        }),
        h("img", { class: "devthumb", alt: "", hidden: true }),
        h("div", { class: "devthumb placeholder" }),
        h("div", { class: "grow" },
          h("div", { class: "name" }, dev.name),
          h("div", { class: "row", style: { gap: "6px", marginTop: "4px" } },
            h("span", { class: "tag" }, `index ${dev.index}`),
            dev.virtual ? h("span", { class: "tag" }, "virtual") : null,
            dev.available === true ? h("span", { class: "tag ok" }, "working") : null,
            dev.available === false ? h("span", { class: "tag bad" }, "not responding") : null,
            h("span", { class: "faint state" })),
          dev.detail ? h("div", { class: "faint" }, dev.detail) : null),
        h("select", {
          onChange: (e) => update(dev.index, dev.name, { role: e.target.value }),
        },
          h("option", { value: "side" }, "Side view"),
          h("option", { value: "front" }, "Front view")))));

    for (const dev of devices) {
      const row = list.querySelector(`[data-index="${dev.index}"]`);
      if (!row) continue;
      const cam = configured.find((c) => c.index === dev.index);
      const live = running.find((c) => c.index === dev.index);
      const box = row.querySelector("input");
      const select = row.querySelector("select");
      if (document.activeElement !== box) box.checked = !!(cam && cam.enabled);
      if (document.activeElement !== select) select.value = cam ? cam.role : "side";
      row.querySelector(".state").textContent = live ? live.state_label : "";

      const img = row.querySelector("img.devthumb");
      const placeholder = row.querySelector(".devthumb.placeholder");
      const wantThumb = !!live && previewOn;
      img.hidden = !wantThumb;
      placeholder.hidden = wantThumb;
      // Thumbnails refresh slowly on purpose: this screen is for setting
      // cameras up, not for watching them.
      if (wantThumb && Date.now() - (thumbAt.get(dev.index) || 0) > 1000) {
        thumbAt.set(dev.index, Date.now());
        img.src = `/api/frame/${dev.index}?w=192&t=${Date.now()}`;
      }
    }
  }

  render();
  store.subscribe(render);
  return root;
}
