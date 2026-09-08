import { h, mount, fmt, toneForRatio } from "../dom.js";
import { PostureOverlay } from "./posture-overlay.js";

const TONE = {
  tracking: "ok", partial: "ok", cannot_see: "warn",
  no_person: "", error: "bad", stopped: "", starting: "",
};

/**
 * One camera in the workspace: image or skeleton, with the posture overlay.
 *
 * Shared by the single-camera stage and the grid, so both get the same request
 * pacing, the same backoff and the same overlay rather than two nearly-equal
 * implementations that drift.
 *
 * @param index   camera index, or null to follow whatever is primary
 * @param onPick  called when the tile is clicked (grid mode promotes it)
 * @param chrome  show a per-tile header — wanted in the grid, redundant in single
 */
export function CameraTile(store, { index = null, onPick = null, chrome = false } = {}) {
  const img = h("img", { alt: "Camera preview" });
  const overlay = PostureOverlay();
  const frame = h("div", { class: "tile-frame" }, img, overlay.node);
  const empty = h("div", { class: "stage-empty" });
  const readout = h("div", { class: "overlay-readout" });
  const head = h("div", { class: "tile-head" });

  const root = h("div", {
    class: `tile ${onPick ? "pickable" : ""}`,
    onClick: onPick ? () => { const c = camera(); if (c) onPick(c.index); } : null,
  }, chrome ? head : null, h("div", { class: "tile-body" }, frame, empty, readout));

  let lastIndex = null;
  let inFlight = false;
  // The preview endpoint honestly 404s until a camera produces its first
  // frame. Retrying twice a second through that window achieves nothing and
  // fills the console, so back off and say what is happening.
  let failures = 0;
  let skip = 0;
  let everLoaded = false;

  img.addEventListener("load", () => { inFlight = false; failures = 0; everLoaded = true; });
  img.addEventListener("error", () => {
    inFlight = false;
    failures += 1;
    skip = Math.min(10, failures);
  });

  function camera() {
    if (index === null) return store.primary();
    return ((store.status && store.status.cameras) || [])
      .find((c) => c.index === index) || null;
  }

  function request(cam, width) {
    inFlight = true;
    img.src = `/api/frame/${cam.index}?w=${width}&t=${Date.now()}`;
  }

  function render() {
    const cam = camera();
    const cfg = store.config;
    const mode = (cfg && cfg.web && cfg.web.preview_mode) || "video";
    const thresh = cfg ? cfg.sampling.visibility_threshold : 0.6;
    const primary = store.primary();

    root.classList.toggle("active", !!(primary && cam && primary.index === cam.index));

    if (!cam) {
      frame.hidden = true; readout.hidden = true; empty.hidden = false;
      empty.textContent = store.connected
        ? "No camera running. Enable one under Cameras."
        : "Waiting for the monitor…";
      if (chrome) mount(head, h("span", { class: "muted" }, "No camera"));
      return;
    }

    if (chrome) {
      mount(head,
        h("span", { class: `led ${TONE[cam.state] || ""}` }),
        h("span", { class: "nm" }, cam.name),
        h("span", { class: "tag" }, cam.role === "front" ? "Front" : "Side"),
        h("span", { class: "spacer", style: { flex: "1" } }),
        h("span", { class: "faint" }, cam.state_label));
    }

    if (cam.size && cam.size[0] && cam.size[1]) {
      frame.style.setProperty("--ar", `${cam.size[0]} / ${cam.size[1]}`);
    }
    frame.classList.toggle("skeleton", mode !== "video");

    if (mode === "off") {
      frame.hidden = true; readout.hidden = true; empty.hidden = false;
      empty.textContent = "Preview is off. Change it under Settings.";
      return;
    }

    if (mode === "skeleton") {
      // No image is requested at all; the server is not keeping frames in this
      // mode. The overlay alone shows framing and posture.
      img.removeAttribute("src");
      empty.hidden = true; frame.hidden = false;
    } else if (failures > 2 && !everLoaded) {
      frame.hidden = true; readout.hidden = true; empty.hidden = false;
      empty.textContent = cam.state === "starting"
        ? "Starting the camera…" : "Waiting for the first frame…";
      if (skip > 0) skip -= 1; else if (!inFlight) request(cam, 480);
      return;
    } else {
      empty.hidden = true; frame.hidden = false;
      if (cam.index !== lastIndex) {
        lastIndex = cam.index; failures = 0; skip = 0; everLoaded = false;
      }
      // Ask for roughly the size it will be shown at, rounded so the URL
      // repeats between ticks instead of making the server re-encode at a new
      // width on every window nudge.
      const want = Math.min(960, Math.max(240, Math.round(
        (frame.clientWidth || 480) * (window.devicePixelRatio || 1) / 64) * 64));
      if (skip > 0) skip -= 1; else if (!inFlight) request(cam, want);
    }

    const posture = (store.status && store.status.posture) || {};
    overlay.render(cam, posture, thresh, { skeletonOnly: mode === "skeleton" });

    const shown = chrome ? [] : (posture.metrics || []).slice(0, 3);
    readout.hidden = shown.length === 0;
    mount(readout, shown.map((m) => h("div", { class: "row" },
      h("span", { class: "k" }, m.label),
      h("span", { class: `v num ${toneForRatio(m.ratio)}` }, fmt(m.value, m.unit)))));
  }

  render();
  store.subscribe(render);
  return root;
}
