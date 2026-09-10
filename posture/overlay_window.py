"""The full-screen overlay, and the hold-to-clear mechanic made visible.

This is the escalation's last step and the only part of the app that takes over
your screen, so it is built to be honest about how to get rid of it: a live
countdown of the hold, driven by the same posture readings as everything else.
It cannot be dismissed by clicking it away. The only exits are fixing your
posture for the required stretch, or an explicit, time-boxed snooze -- which
Escape triggers, so the exit does not require a mouse.

Threading
---------
Tk is not thread-safe and wants its own loop, so the window owns a thread and
everything Tk-related happens on it. The rest of the app talks to it only
through :meth:`show`, :meth:`update`, and :meth:`hide`, which just set fields;
the loop picks them up on its own schedule. That keeps the capture and web
threads free of Tk entirely.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

BG = "#0B0F0D"
INK = "#E8EDE9"
DIM = "#8A9992"
BAD = "#F06A5D"
GOOD = "#4ADE9B"
WARN = "#E8B84B"
LINE = "#26332C"


# If nothing updates the overlay for this long, it takes itself down. A window
# that blocks the whole screen must not be able to outlive the data justifying
# it: unplug the camera while it is up and, without this, it would sit there
# forever with no posture reading able to dismiss it.
STALE_AFTER_S = 20.0


@dataclass(frozen=True)
class OverlayView:
    """The live figure, and why the hold is or is not counting down.

    Landmark positions only, never a frame. This window covers the whole
    screen, so putting the camera picture on it would paint a live video of the
    room over whatever is being screen-shared at the time. Positions are enough
    to align by, and they are the same derived numbers the panel already gets.

    ``state`` and ``reason`` come straight from the detector. They are here
    because the countdown alone is ambiguous: a hold that is not advancing
    looks identical whether your posture is wrong or the camera simply cannot
    see the landmarks it needs, and only one of those is fixed by sitting up.
    """

    landmarks: tuple = ()
    thresh: float = 0.5
    aspect: float = 4 / 3
    state: str = "unknown"
    reason: str = ""

    @property
    def measuring(self) -> bool:
        """Whether this reading can move the hold at all."""
        return self.state in ("good", "bad")


# Indices into the landmark payload the monitor publishes
# (monitor.PREVIEW_LANDMARKS), not raw MediaPipe landmark numbers.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_HIP, R_HIP, L_KNEE, R_KNEE = 9, 10, 11, 12

# Drawn faintly, to carry the framing. Without a picture underneath, a bare
# head-and-shoulders line is hard to read as a body at a glance.
LIMBS = ((L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW),
         (L_HIP, L_KNEE), (R_HIP, R_KNEE),
         (L_EAR, L_EYE), (R_EAR, R_EYE), (L_EYE, NOSE), (R_EYE, NOSE))


def figure(landmarks, thresh: float) -> list[tuple]:
    """The seated figure as primitives in normalized 0..1 coordinates.

    Pure geometry, kept out of the Tk thread so it can be tested without a
    display. Mirrors what the panel draws (``posture-overlay.js``): the
    measured segments in full strength, the limbs faint, and a dotted vertical
    through the shoulders -- the reference every angle in the app is taken
    from, and the thing you are actually trying to line up with.

    Deliberately not mirrored. The panel shows the camera's own view, and a
    figure that flipped between the two windows would be worse than one that is
    merely back-to-front.
    """
    out: list[tuple] = []
    if not landmarks:
        return out

    def seen(i: int) -> bool:
        return i < len(landmarks) and landmarks[i][2] >= thresh

    def pt(i: int) -> tuple[float, float]:
        return landmarks[i][0], landmarks[i][1]

    def mid(a: int, b: int):
        """Midpoint of whichever of the pair is actually visible."""
        if seen(a) and seen(b):
            ax, ay = pt(a)
            bx, by = pt(b)
            return (ax + bx) / 2, (ay + by) / 2
        if seen(a):
            return pt(a)
        if seen(b):
            return pt(b)
        return None

    shoulder = mid(L_SHOULDER, R_SHOULDER)
    ear = mid(L_EAR, R_EAR)
    hip = mid(L_HIP, R_HIP)

    if shoulder is not None:
        sx, sy = shoulder
        out.append(("line", sx, max(0.0, sy - 0.34), sx, min(1.0, sy + 0.26), "ref"))

    for a, b in LIMBS:
        if seen(a) and seen(b):
            ax, ay = pt(a)
            bx, by = pt(b)
            out.append(("line", ax, ay, bx, by, "faint"))

    if seen(L_SHOULDER) and seen(R_SHOULDER):
        ax, ay = pt(L_SHOULDER)
        bx, by = pt(R_SHOULDER)
        out.append(("line", ax, ay, bx, by, "key"))

    # Ear-to-shoulder is the segment neck tilt measures, and the one metric
    # that survives with no hips in frame -- which at a desk is most of the
    # time. It is drawn whenever it can be.
    if ear is not None and shoulder is not None:
        out.append(("line", ear[0], ear[1], shoulder[0], shoulder[1], "key"))
        out.append(("dot", ear[0], ear[1], "key"))

    # Shoulder-to-hip only when hips are genuinely visible. Drawing a line to a
    # landmark the app is ignoring would imply a measurement that is not being
    # taken.
    if hip is not None and shoulder is not None:
        out.append(("line", shoulder[0], shoulder[1], hip[0], hip[1], "key"))
        out.append(("dot", hip[0], hip[1], "key"))

    if shoulder is not None:
        out.append(("dot", shoulder[0], shoulder[1], "key"))

    for i, lm in enumerate(landmarks):
        if lm[2] < 0.05:
            continue
        out.append(("dot", lm[0], lm[1], "joint" if lm[2] >= thresh else "lost"))
    return out


FIGURE_W, FIGURE_H = 460, 360

_FIGURE_COLOURS = {
    "faint": "#38473F",
    "ref": "#4A5B52",
    "joint": DIM,
    "lost": "#7A3B36",
}


def _accent(state: str) -> str:
    """The one colour that says whether the hold is advancing."""
    if state == "good":
        return GOOD
    if state == "bad":
        return BAD
    return WARN


def hold_message(view: "OverlayView", remaining: float | None) -> tuple[str, str]:
    """The hold readout: what it says, and the colour of the progress bar.

    A hold that is not advancing has two very different causes, and a frozen
    countdown cannot tell them apart: either you are sitting badly, or nothing
    is being measured at all and sitting up will achieve nothing until you are
    back in view. Saying which is the difference between a window you can
    dismiss and one that feels stuck.
    """
    if not view.measuring:
        # Landmark names arrive as identifiers. The headline already spells
        # offenders out, and "right_shoulder" on a screen-filling window reads
        # as a stack trace rather than a body part.
        reason = (view.reason or "not measuring you").replace("_", " ")
        return (reason[0].upper() + reason[1:]
                + " — the hold is paused until you are back in view", WARN)
    if view.state == "good" and remaining is not None:
        return f"Holding — {remaining:.0f}s to go", GOOD
    return "Sit back to your calibrated posture to start the hold", BAD


def _render_figure(canvas, view: "OverlayView", box_w: int, box_h: int) -> None:
    """Draw the figure into a Tk canvas. Called only on the Tk thread."""
    canvas.delete("all")

    # The frame's own edges, so being half out of shot reads as being half out
    # of shot rather than as a body that has lost an arm.
    aspect = view.aspect if view.aspect and view.aspect > 0 else 4 / 3
    if aspect >= box_w / box_h:
        w = float(box_w)
        h = w / aspect
    else:
        h = float(box_h)
        w = h * aspect
    ox, oy = (box_w - w) / 2, (box_h - h) / 2
    canvas.create_rectangle(ox, oy, ox + w, oy + h, outline=LINE, width=1)

    prims = figure(view.landmarks, view.thresh)
    if not prims:
        canvas.create_text(box_w / 2, box_h / 2, text="nothing in view",
                           fill=DIM, font=("Segoe UI", 11))
        return

    accent = _accent(view.state)

    def px(nx: float, ny: float) -> tuple[float, float]:
        return ox + nx * w, oy + ny * h

    for prim in prims:
        kind = prim[-1]
        colour = accent if kind == "key" else _FIGURE_COLOURS[kind]
        if prim[0] == "line":
            x1, y1 = px(prim[1], prim[2])
            x2, y2 = px(prim[3], prim[4])
            if kind == "ref":
                canvas.create_line(x1, y1, x2, y2, fill=colour, width=1,
                                   dash=(3, 4))
            else:
                canvas.create_line(x1, y1, x2, y2, fill=colour,
                                   width=3 if kind == "key" else 2,
                                   capstyle="round")
        else:
            cx, cy = px(prim[1], prim[2])
            r = 4.0 if kind == "key" else 2.5
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=colour,
                               outline="")


class OverlayWindow:
    """A borderless, always-on-top, full-screen nag with a hold countdown."""

    def __init__(self, on_snooze=None, stale_after: float = STALE_AFTER_S) -> None:
        self._on_snooze = on_snooze
        self._stale_after = stale_after
        self._last_update = 0.0
        self._commands: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._visible = False
        self._available: bool | None = None

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def available(self) -> bool:
        """Whether a window can be created at all (a headless box cannot)."""
        if self._available is None:
            try:
                import tkinter  # noqa: F401
                self._available = True
            except Exception:
                log.warning("tkinter unavailable; the overlay step is disabled")
                self._available = False
        return self._available

    # -- public API (any thread) -------------------------------------------

    def show(self, headline: str, detail: str, hold_required: float,
             hold_remaining: float | None, view: "OverlayView | None" = None) -> None:
        if not self.available:
            return
        self._visible = True
        self._last_update = time.monotonic()
        self._ensure_thread()
        self._commands.put(("show", headline, detail, hold_required, hold_remaining,
                            view or OverlayView()))

    def update(self, headline: str, detail: str, hold_required: float,
               hold_remaining: float | None, view: "OverlayView | None" = None) -> None:
        if not self._visible:
            return
        self._last_update = time.monotonic()
        self._commands.put(("update", headline, detail, hold_required, hold_remaining,
                            view or OverlayView()))

    def hide(self) -> None:
        if not self._visible:
            return
        self._visible = False
        self._commands.put(("hide", "", "", 0.0, None, OverlayView()))

    def press(self, key: str) -> None:
        """Deliver a keypress to the window. For tests: the overlay owns its
        own Tk thread, so a test cannot reach the widget directly."""
        if not self._visible:
            return
        self._commands.put(("key", key, "", 0.0, None, OverlayView()))

    def stop(self, timeout: float = 3.0) -> None:
        """Close the window and wait for its thread to finish.

        The wait matters: the Tk interpreter was created on that thread and has
        to be torn down there too. Letting the process exit while it is still
        alive leaves the interpreter to be finalised from the main thread,
        which prints "Tcl_AsyncDelete: async handler deleted by the wrong
        thread" on the way out.
        """
        self._visible = False
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        self._commands.put(("stop", "", "", 0.0, None, OverlayView()))
        thread.join(timeout=timeout)
        self._thread = None

    # -- the Tk thread -----------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="overlay", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import tkinter as tk

        try:
            root = tk.Tk()
        except Exception:
            log.warning("could not open the overlay window", exc_info=True)
            self._available = False
            return

        root.withdraw()
        root.title("Posture")
        root.configure(bg=BG)
        root.attributes("-topmost", True)
        # No close button and no window manager decoration: this is not a
        # dialog to be dismissed, and a titlebar X would suggest otherwise.
        #
        # Sized to the screen by hand rather than with the -fullscreen
        # attribute, which Tk refuses to combine with override-redirect
        # ("can't set fullscreen attribute: override-redirect flag is set").
        # Manual geometry gets the same coverage and keeps the window out of
        # the window manager's reach.
        root.overrideredirect(True)
        root.protocol("WM_DELETE_WINDOW", lambda: None)

        wrap = tk.Frame(root, bg=BG)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        headline = tk.Label(wrap, text="", bg=BG, fg=BAD,
                            font=("Segoe UI", 40, "normal"))
        headline.pack(pady=(0, 10))
        detail = tk.Label(wrap, text="", bg=BG, fg=DIM,
                          font=("Segoe UI", 14), wraplength=760, justify="center")
        detail.pack(pady=(0, 18))

        # What the camera can see of you right now. The countdown says how much
        # longer to hold; this says whether you are even in a position to be
        # measured, which is the part that was impossible to guess.
        canvas = tk.Canvas(wrap, width=FIGURE_W, height=FIGURE_H, bg=BG,
                           highlightthickness=0)
        canvas.pack(pady=(0, 20))

        hold_label = tk.Label(wrap, text="", bg=BG, fg=DIM,
                              font=("Segoe UI", 11))
        hold_label.pack(pady=(0, 8))
        bar_w, bar_h = 460, 6
        bar = tk.Canvas(wrap, width=bar_w, height=bar_h, bg=LINE,
                        highlightthickness=0)
        bar.pack()
        fill = bar.create_rectangle(0, 0, 0, bar_h, fill=GOOD, width=0)

        note = tk.Label(wrap, text="This closes when you have held a good posture.",
                        bg=BG, fg="#5D6B64", font=("Segoe UI", 10))
        note.pack(pady=(26, 10))

        snooze = tk.Button(
            wrap, text="Snooze  ·  Esc", bg="#161E1A", fg=INK,
            activebackground="#1B241F",
            activeforeground=INK, relief="flat", padx=18, pady=7,
            font=("Segoe UI", 10), cursor="hand2",
            command=lambda: self._snooze_clicked(),
        )
        snooze.pack()

        # Escape does exactly what the button does, and it is on the button so
        # that it can be found. This is not a new way out -- the snooze was
        # always one click away -- it is the same way out reachable without a
        # mouse. A window that covers the screen and can only be dismissed by
        # pointing at it is a trap for anyone whose hands are on the keyboard,
        # which at a desk is everyone.
        root.bind("<Escape>", lambda _e: self._snooze_clicked())

        def pump() -> None:
            try:
                while True:
                    (kind, head, det, required, remaining,
                     view) = self._commands.get_nowait()
                    if kind == "stop":
                        root.quit()
                        root.destroy()
                        return
                    if kind == "hide":
                        root.withdraw()
                        continue
                    if kind == "key":
                        root.event_generate(head)
                        continue
                    headline.config(text=head)
                    detail.config(text=det)
                    _render_figure(canvas, view, FIGURE_W, FIGURE_H)
                    if required > 0 and remaining is not None:
                        done = max(0.0, min(1.0, 1.0 - remaining / required))
                        bar.coords(fill, 0, 0, bar_w * done, bar_h)
                    else:
                        bar.coords(fill, 0, 0, 0, bar_h)

                    text, colour = hold_message(view, remaining)
                    bar.itemconfig(fill, fill=colour)
                    hold_label.config(text=text,
                                      fg=WARN if not view.measuring else DIM)
                    if kind == "show":
                        root.geometry(
                            f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
                        root.deiconify()
                        root.attributes("-topmost", True)
                        root.lift()
                        # An override-redirect window gets no keyboard focus
                        # from Windows, so Escape would never reach it. Taking
                        # focus is the trade this makes: keystrokes typed at a
                        # screen you cannot see now land here and are dropped,
                        # rather than going invisibly into whatever is behind.
                        # Given the window is already covering that, dropping
                        # them is the better of the two.
                        try:
                            root.focus_force()
                        except Exception:  # pragma: no cover - platform quirk
                            log.debug("could not focus the overlay", exc_info=True)
            except queue.Empty:
                pass

            # Safety valve: nothing has told us anything for a while, so the
            # readings that raised this window are gone. Take it down rather
            # than block the screen on stale information.
            if (self._visible and self._last_update
                    and time.monotonic() - self._last_update > self._stale_after):
                log.warning("overlay went stale after %.0fs with no update; hiding",
                            self._stale_after)
                self._visible = False
                root.withdraw()

            root.after(120, pump)

        root.after(120, pump)
        try:
            root.mainloop()
        except Exception:  # pragma: no cover - teardown best effort
            log.debug("overlay loop ended", exc_info=True)
        finally:
            # Drop every reference to the interpreter on the thread that made
            # it, then collect. Left to the main thread at process exit, Tk
            # complains: "Tcl_AsyncDelete: async handler deleted by the wrong
            # thread". The widgets are locals, but the callbacks close over
            # them, so the cycle needs a collection to break here and not later.
            del wrap, headline, detail, canvas, hold_label, bar, note, snooze, root
            gc.collect()

    def _snooze_clicked(self) -> None:
        if self._on_snooze is not None:
            try:
                self._on_snooze()
            except Exception:
                log.exception("snooze from the overlay failed")
        self.hide()
