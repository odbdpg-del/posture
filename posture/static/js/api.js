// Every call the panel makes to the local server, in one place. Nothing here
// knows about the DOM; nothing elsewhere builds a URL.

async function json(url, options) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok && body.error) throw new Error(body.error);
  return body;
}

const post = (url, payload) => json(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload || {}),
});

export const api = {
  status: () => json("/api/status"),
  config: () => json("/api/config"),
  devices: (check) => json("/api/devices" + (check ? "?check=1" : "")),
  stats: (day) => json("/api/stats" + (day ? `?day=${day}` : "")),
  statsDays: () => json("/api/stats/days"),
  history: (windowSeconds) =>
    json("/api/history" + (windowSeconds ? `?window=${windowSeconds}` : "")),
  saveConfig: (cfg) => post("/api/config", cfg),
  calibrate: (camera) => post("/api/calibrate", camera == null ? {} : { camera }),
  cancelCalibration: () => post("/api/calibrate/cancel"),
  clearBaselines: () => post("/api/calibrate/clear"),
  snooze: (seconds) => post("/api/snooze", seconds == null ? {} : { seconds }),
  cancelSnooze: () => post("/api/snooze/cancel"),
  quit: () => post("/api/quit"),
  frameUrl: (index) => `/api/frame/${index}?t=${Date.now()}`,
};
