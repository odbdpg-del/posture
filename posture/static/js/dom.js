// Minimal DOM helpers. No framework: the panel is served from a loopback
// Python process with a strict CSP and no build step, so a bundler and a
// runtime would be three new moving parts to buy very little.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "style") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") {
      el.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (k === "text") el.textContent = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);

/**
 * Rebuild `container` only when `signature` changes.
 *
 * The store emits about twice a second. Anything that calls mount() on every
 * emit replaces its own nodes underneath the cursor, and a click that starts
 * before an emit and ends after it lands on an element that no longer exists —
 * which is exactly the "button did nothing the first time" symptom. So
 * anything holding a control rebuilds on a signature and updates volatile text
 * in place.
 *
 * Returns true when a rebuild happened, so callers can do first-time wiring.
 */
export function reconcile(container, signature, build) {
  if (container.dataset.sig === signature) return false;
  container.dataset.sig = signature;
  mount(container, build());
  return true;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function mount(parent, ...nodes) {
  clear(parent);
  parent.append(...nodes.flat().filter(Boolean));
  return parent;
}

/** Seconds as H:MM:SS or M:SS — used for session time and alert durations. */
export function duration(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "--";
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** Compact "8m 32s" phrasing for prose contexts. */
export function shortDuration(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "--";
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function clockTime(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour: "numeric", minute: "2-digit",
  });
}

/** Format a metric value with its unit, keeping a sign for deviations. */
export function fmt(value, unit, digits = 1) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  const sign = n > 0 ? "+" : "";
  const u = unit === "deg" ? "°" : ` ${unit}`;
  return `${sign}${n.toFixed(digits)}${u}`;
}

/** Posture tone for a ratio of tolerance: <0.5 good, <1 watch, else correct. */
export function toneForRatio(ratio) {
  if (ratio === null || ratio === undefined) return "idle";
  if (ratio >= 1) return "bad";
  if (ratio >= 0.5) return "warn";
  return "good";
}

export const STATUS_WORD = { good: "GOOD", warn: "WATCH", bad: "CORRECT", idle: "—" };
