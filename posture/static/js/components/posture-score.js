import { h, mount, shortDuration } from "../dom.js";

/**
 * The one number the whole panel is arranged around.
 *
 * Shows nothing rather than something reassuring when there is nobody there or
 * no baseline: a big green 100 over an empty chair would be a lie, and the
 * server deliberately sends null in those cases.
 */
export function PostureScore(store) {
  const root = h("section", { class: "panel" },
    h("div", { class: "panel-head" }, "Posture"),
    h("div", { class: "panel-body", style: { padding: "0" } }));
  const body = root.lastChild;

  function render() {
    const s = store.status || {};
    const score = s.score || {};
    const has = score.value !== null && score.value !== undefined;

    mount(body, h("div", { class: `score ${score.tone || "idle"}` },
      h("div", { class: "value num" }, has ? String(score.value) : "—"),
      h("div", { class: "band" }, has ? score.band : (score.band || "no reading")),
      h("div", { class: "sub" },
        has
          ? (s.state_since !== undefined
              ? `Stable for ${shortDuration(s.state_since)}`
              : "")
          : (score.reason || "")),
      has && score.worst
        ? h("div", { class: "sub" }, `Held back by ${score.worst.toLowerCase()}`)
        : null,
      h("div", { class: "avg" },
        h("span", null, "Session average"),
        h("span", { class: "num" },
          s.session_average === null || s.session_average === undefined
            ? "—" : String(s.session_average))),
    ));
  }

  render();
  store.subscribe(render);
  return root;
}
