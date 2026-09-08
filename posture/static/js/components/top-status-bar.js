import { h, mount, duration } from "../dom.js";

// Maps a camera's worker state onto the three-way health colour used for LEDs.
const CAM_TONE = {
  tracking: "ok", partial: "ok", cannot_see: "warn",
  no_person: "", error: "bad", stopped: "", starting: "",
};

export function TopStatusBar(store, onQuit) {
  const root = h("header", { class: "topbar" });

  function render() {
    const s = store.status;
    const posture = (s && s.posture) || {};
    const running = s && s.running;

    let tone = "", word = "Connecting";
    if (!store.connected) { tone = "bad"; word = "Disconnected"; }
    else if (s && s.shutting_down) { word = "Stopped"; }
    else if (running) {
      word = "Monitoring";
      tone = posture.state === "bad" ? "bad"
        : posture.state === "unknown" ? "warn" : "live";
    } else { word = "Idle"; }

    const cams = (s && s.cameras) || [];
    mount(root,
      h("span", { class: "brand" }, "POSTURE"),
      h("span", { class: "sep" }),
      h("span", { class: `statechip ${tone}` }, h("span", { class: "led" }), word),
      h("span", { class: "sep" }),
      h("span", { class: "label" }, "Session"),
      h("span", { class: "sessiontime num" }, duration(s && s.uptime)),
      h("span", { class: "spacer" }),
      h("div", { class: "camchips" }, cams.map((c) =>
        h("span", {
          class: `camchip ${CAM_TONE[c.state] || ""}`,
          title: `${c.name} — ${c.state_label}`,
        }, h("span", { class: "led" }), shortName(c.name)))),
      h("span", { class: "sep" }),
      h("button", { class: "danger sm", onClick: onQuit }, "Stop monitoring"),
    );
  }

  render();
  store.subscribe(render);
  return root;
}

// "Logitech BRIO" -> "BRIO". Device names are long and the header is not.
function shortName(name) {
  if (!name) return "camera";
  const words = String(name).split(/\s+/);
  return (words.length > 1 ? words[words.length - 1] : words[0]).toUpperCase();
}
