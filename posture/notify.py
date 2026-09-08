"""OS notifications, with a fallback that always works.

Two backends, chosen at startup:

* ``winotify`` builds a real Windows toast, which lands in the Action Centre
  and behaves like every other notification on the machine. Preferred.
* A Tk banner in the corner of the screen, for when that is unavailable.

The fallback exists because a posture reminder that silently fails to appear is
worse than a plain one that shows up. Both are local; nothing here talks to a
network or writes anything down.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

APP_ID = "Posture"


class Notifier:
    """Sends a notification, by whatever route works on this machine."""

    def __init__(self, app_name: str = APP_ID) -> None:
        self.app_name = app_name
        self._backend = self._pick_backend()

    @property
    def backend(self) -> str:
        return self._backend

    def _pick_backend(self) -> str:
        try:
            import winotify  # noqa: F401
        except Exception:
            return "banner"
        return "toast"

    def send(self, title: str, message: str) -> bool:
        """Show one notification. Never raises; returns whether it went out."""
        try:
            if self._backend == "toast" and self._toast(title, message):
                return True
        except Exception:
            log.debug("toast failed, falling back to a banner", exc_info=True)
        try:
            return self._banner(title, message)
        except Exception:
            log.warning("could not show a notification", exc_info=True)
            return False

    def _toast(self, title: str, message: str) -> bool:
        from winotify import Notification

        toast = Notification(app_id=self.app_name, title=title, msg=message)
        toast.show()
        return True

    def _banner(self, title: str, message: str, seconds: float = 6.0) -> bool:
        """A small always-on-top banner, bottom-right, that fades itself out.

        Runs its own Tk root on its own thread, so it cannot interfere with the
        full-screen overlay's event loop or with anything else in the process.
        """
        def run() -> None:
            import tkinter as tk

            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.configure(bg="#111714")
            try:
                root.attributes("-alpha", 0.96)
            except tk.TclError:
                pass

            frame = tk.Frame(root, bg="#111714", padx=16, pady=12,
                             highlightbackground="#26332C", highlightthickness=1)
            frame.pack(fill="both", expand=True)
            tk.Label(frame, text=title, bg="#111714", fg="#E8EDE9",
                     font=("Segoe UI", 11, "bold"), anchor="w",
                     justify="left").pack(anchor="w")
            tk.Label(frame, text=message, bg="#111714", fg="#8A9992",
                     font=("Segoe UI", 9), anchor="w", justify="left",
                     wraplength=300).pack(anchor="w", pady=(3, 0))

            root.update_idletasks()
            width = max(280, root.winfo_reqwidth())
            height = root.winfo_reqheight()
            x = root.winfo_screenwidth() - width - 24
            y = root.winfo_screenheight() - height - 64
            root.geometry(f"{width}x{height}+{x}+{y}")

            root.after(int(seconds * 1000), root.destroy)
            root.mainloop()

        threading.Thread(target=run, name="notify-banner", daemon=True).start()
        return True
