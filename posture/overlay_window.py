"""The full-screen overlay, and the hold-to-clear mechanic made visible.

This is the escalation's last step and the only part of the app that takes over
your screen, so it is built to be honest about how to get rid of it: a live
countdown of the hold, driven by the same posture readings as everything else.
It cannot be dismissed by clicking. The only exits are fixing your posture for
the required stretch, or an explicit, time-boxed snooze.

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

log = logging.getLogger(__name__)

BG = "#0B0F0D"
INK = "#E8EDE9"
DIM = "#8A9992"
BAD = "#F06A5D"
GOOD = "#4ADE9B"
LINE = "#26332C"


# If nothing updates the overlay for this long, it takes itself down. A window
# that blocks the whole screen must not be able to outlive the data justifying
# it: unplug the camera while it is up and, without this, it would sit there
# forever with no posture reading able to dismiss it.
STALE_AFTER_S = 20.0


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
             hold_remaining: float | None) -> None:
        if not self.available:
            return
        self._visible = True
        self._last_update = time.monotonic()
        self._ensure_thread()
        self._commands.put(("show", headline, detail, hold_required, hold_remaining))

    def update(self, headline: str, detail: str, hold_required: float,
               hold_remaining: float | None) -> None:
        if not self._visible:
            return
        self._last_update = time.monotonic()
        self._commands.put(("update", headline, detail, hold_required, hold_remaining))

    def hide(self) -> None:
        if not self._visible:
            return
        self._visible = False
        self._commands.put(("hide", "", "", 0.0, None))

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
        self._commands.put(("stop", "", "", 0.0, None))
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
        detail.pack(pady=(0, 34))

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
            wrap, text="Snooze", bg="#161E1A", fg=INK, activebackground="#1B241F",
            activeforeground=INK, relief="flat", padx=18, pady=7,
            font=("Segoe UI", 10), cursor="hand2",
            command=lambda: self._snooze_clicked(),
        )
        snooze.pack()

        def pump() -> None:
            try:
                while True:
                    kind, head, det, required, remaining = self._commands.get_nowait()
                    if kind == "stop":
                        root.quit()
                        root.destroy()
                        return
                    if kind == "hide":
                        root.withdraw()
                        continue
                    headline.config(text=head)
                    detail.config(text=det)
                    if required > 0 and remaining is not None:
                        done = max(0.0, min(1.0, 1.0 - remaining / required))
                        bar.coords(fill, 0, 0, bar_w * done, bar_h)
                        hold_label.config(
                            text=f"Hold a good posture — {remaining:.0f}s to go")
                    else:
                        bar.coords(fill, 0, 0, 0, bar_h)
                        hold_label.config(text="Hold a good posture to dismiss")
                    if kind == "show":
                        root.geometry(
                            f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
                        root.deiconify()
                        root.attributes("-topmost", True)
                        root.lift()
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
            del wrap, headline, detail, hold_label, bar, note, snooze, root
            gc.collect()

    def _snooze_clicked(self) -> None:
        if self._on_snooze is not None:
            try:
                self._on_snooze()
            except Exception:
                log.exception("snooze from the overlay failed")
        self.hide()
