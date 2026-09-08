import { h, mount } from "../dom.js";
import { api } from "../api.js";
import { refreshConfig } from "../store.js";
import { CameraTile } from "./camera-tile.js";
import { CameraSelector } from "./camera-selector.js";

/**
 * The camera workspace: one large feed, or a grid of all of them.
 *
 * Single is the default because the hierarchy the app is built around starts
 * with one live person, not a wall of monitors. The grid is there for setting
 * cameras up and for anyone who wants both angles at once, and it is a layout
 * switch rather than a separate section so the surrounding instrumentation
 * stays where it is.
 */
export function LiveCameraView(store) {
  const title = h("div", { class: "camselect" });
  // Built once: these are controls, and rebuilding them each poll would eat
  // the click that started just before it.
  const singleBtn = h("button", {
    class: "segbtn", title: "One camera",
    onClick: () => store.set({ cameraLayout: "single" }),
  }, "Single");
  const gridBtn = h("button", {
    class: "segbtn", onClick: () => store.set({ cameraLayout: "grid" }),
  }, "Grid");
  const layoutToggle = h("div", { class: "seg" }, singleBtn, gridBtn);

  // Preview mode belongs here as well as in Settings: it changes what this
  // panel shows, and burying the only copy three screens away made it
  // effectively undiscoverable.
  const modeSelect = h("select", {
    class: "headsel", title: "What the camera panel shows",
    onChange: async (e) => {
      const cfg = JSON.parse(JSON.stringify(store.config));
      cfg.web.preview_mode = e.target.value;
      const res = await api.saveConfig(cfg);
      if (res.ok) await refreshConfig();
    },
  },
    h("option", { value: "video" }, "Camera video"),
    h("option", { value: "skeleton" }, "Skeleton only"),
    h("option", { value: "off" }, "Preview off"));
  const body = h("div", { class: "panel-body" });
  const selector = CameraSelector(store);

  const root = h("section", { class: "panel stage" },
    h("div", { class: "panel-head" },
      title,
      h("span", { style: { flex: "1" } }),
      modeSelect,
      layoutToggle),
    body);

  // Tiles are built once per camera and kept, so switching layout does not
  // restart every image request.
  const single = CameraTile(store, { index: null, chrome: false });
  const gridTiles = new Map();
  let builtLayout = null;
  let builtKey = "";

  function grid(indices) {
    for (const i of indices) {
      if (!gridTiles.has(i)) {
        gridTiles.set(i, CameraTile(store, {
          index: i, chrome: true,
          onPick: (picked) => store.set({ primaryCamera: picked, cameraLayout: "single" }),
        }));
      }
    }
    return indices.map((i) => gridTiles.get(i));
  }

  function render() {
    const cams = (store.status && store.status.cameras) || [];
    const layout = cams.length > 1 ? store.cameraLayout : "single";
    const cam = store.primary();

    const mode = (store.config && store.config.web && store.config.web.preview_mode)
      || "video";
    if (document.activeElement !== modeSelect) modeSelect.value = mode;

    singleBtn.classList.toggle("on", layout === "single");
    gridBtn.classList.toggle("on", layout === "grid");
    gridBtn.disabled = cams.length < 2;
    gridBtn.title = cams.length < 2
      ? "Needs more than one camera" : "All cameras";

    if (layout === "grid") {
      mount(title, h("span", null, "All cameras"),
        h("span", { class: "faint" }, `${cams.length} running`));
    } else if (cam) {
      mount(title,
        h("span", null, cam.name),
        h("span", { class: "tag" }, cam.role === "front" ? "Front view" : "Side view"),
        h("span", { class: "faint" }, cam.state_label));
    } else {
      mount(title, h("span", { class: "muted" }, "No camera"));
    }

    const key = cams.map((c) => c.index).join(",");
    if (builtLayout !== layout || builtKey !== key) {
      builtLayout = layout; builtKey = key;
      if (layout === "grid") {
        body.className = "panel-body grid-body";
        mount(body, grid(cams.map((c) => c.index)));
      } else {
        body.className = "panel-body";
        mount(body, single, cams.length > 1 ? selector : null);
      }
    }
  }

  render();
  store.subscribe(render);
  return root;
}
