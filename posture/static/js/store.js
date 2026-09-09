// Application state and the polling that feeds it.
//
// One store, one poll loop, one place that talks to the API. Components read
// state and subscribe to changes; they never fetch. That keeps the posture
// pipeline (server) and its presentation (here) cleanly separated, and means a
// component can be re-rendered from a snapshot without side effects.

import { api } from "./api.js";

const STATUS_MS = 500;    // live readouts
const HISTORY_MS = 5000;  // timeline moves slowly; polling it fast is waste

export const store = {
  view: "live",
  sidebarCollapsed: false,
  status: null,
  config: null,
  devices: null,
  history: null,
  primaryCamera: null,   // index of the camera shown in the stage
  cameraLayout: "single",  // "single" | "grid"
  // How much biomechanical detail the camera overlays draw. See
  // posture-overlay.js MODES. Kept here rather than in the server config
  // because it is a way of looking, not a way of measuring: it changes
  // nothing about what is recorded.
  overlayMode: localStorage.getItem("posture.overlayMode") || "angles",
  connected: false,
  error: "",
  // Camera edits are staged here until applied, so a poll cannot overwrite
  // half-finished changes and the user can see there is something pending.
  pendingCameras: null,
  pendingTolerances: null,

  _subs: new Set(),
  subscribe(fn) { this._subs.add(fn); return () => this._subs.delete(fn); },
  emit() { for (const fn of this._subs) fn(this); },

  set(patch) {
    Object.assign(this, patch);
    if ("overlayMode" in patch) {
      try { localStorage.setItem("posture.overlayMode", patch.overlayMode); } catch { /* private mode */ }
    }
    this.emit();
  },

  /** Cameras as configured, including any unapplied edits. */
  cameras() {
    if (this.pendingCameras) return this.pendingCameras;
    return (this.config && this.config.cameras) || [];
  },

  /** The camera whose feed fills the stage.
   *
   * An explicit pick always wins. Otherwise prefer one that is actually
   * tracking: defaulting to index order put a camera reporting "cannot see you
   * properly" on the main stage while a working one sat in a chip.
   */
  primary() {
    const running = (this.status && this.status.cameras) || [];
    if (!running.length) return null;
    const chosen = running.find((c) => c.index === this.primaryCamera);
    if (chosen) return chosen;
    const rank = { tracking: 0, partial: 1, cannot_see: 2, no_person: 3,
                   starting: 4, stopped: 5, error: 6 };
    return running.slice().sort(
      (a, b) => (rank[a.state] ?? 9) - (rank[b.state] ?? 9))[0];
  },
};

async function pollStatus() {
  try {
    const status = await api.status();
    store.set({ status, connected: true, error: "" });
    if (status.shutting_down) return false;
  } catch (err) {
    store.set({ connected: false, error: String(err.message || err) });
  }
  return true;
}

async function pollHistory() {
  try {
    store.set({ history: await api.history(1800) });
  } catch { /* the timeline simply stays as it was */ }
}

export async function refreshConfig() {
  store.set({ config: await api.config() });
}

export async function refreshDevices(check = false) {
  store.set({ devices: await api.devices(check) });
}

export async function start() {
  await Promise.allSettled([refreshConfig(), refreshDevices(false)]);
  const tickStatus = async () => {
    if (await pollStatus()) setTimeout(tickStatus, STATUS_MS);
  };
  const tickHistory = async () => {
    await pollHistory();
    setTimeout(tickHistory, HISTORY_MS);
  };
  tickStatus();
  tickHistory();
}
