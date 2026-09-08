"""Camera discovery.

Finding out what cameras exist should be instant, and by brute force it is not.
OpenCV has no enumeration API, so the obvious approach is to open indices 0..N
and see which ones answer. Measured on this machine that is unusable: opening
the USB camera through the Media Foundation backend takes 67 seconds, against
1.1 seconds through DirectShow, so probing five indices across three backends
appears to hang for minutes and then reports nothing.

So enumeration and availability are separated:

* :func:`list_devices` asks DirectShow for the device list. It is instant, it
  never opens a camera (no privacy light, no contention with whatever is using
  it), and it returns real product names. The indices it reports are the same
  ones OpenCV uses for its DirectShow backend.
* :func:`check_available` actually opens one camera, and is only called when
  something needs to know whether a specific device works right now.

If DirectShow enumeration is unavailable -- another platform, or pygrabber not
installed -- we fall back to probing, and say so, so the UI can explain why the
list looks thin rather than silently showing nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Virtual cameras are real devices but they are not looking at you; they will
# fail to open unless their host application is running. Worth flagging in the
# UI rather than presenting as a broken webcam.
VIRTUAL_HINTS = ("obs", "virtual", "droidcam", "epoccam", "ndi", "xsplit", "manycam")


@dataclass(frozen=True)
class CameraDevice:
    """One camera the system reports, with whatever we know about it."""

    index: int
    name: str
    available: bool | None = None  # None means "not checked"
    detail: str = ""
    backend: str | None = None
    size: tuple[int, int] | None = None

    @property
    def virtual(self) -> bool:
        low = self.name.lower()
        return any(hint in low for hint in VIRTUAL_HINTS)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "backend": self.backend,
            "size": list(self.size) if self.size else None,
            "virtual": self.virtual,
        }


@dataclass
class Discovery:
    """Result of enumerating cameras, plus how we managed it."""

    devices: list[CameraDevice] = field(default_factory=list)
    method: str = "none"
    warning: str = ""

    def to_dict(self) -> dict:
        return {
            "devices": [d.to_dict() for d in self.devices],
            "method": self.method,
            "warning": self.warning,
        }


def _enumerate_directshow() -> list[CameraDevice] | None:
    """Device names straight from DirectShow, in OpenCV's index order."""
    try:
        # comtypes chatters at INFO about its generated-code cache on every
        # call; that is not news the user needs on stdout.
        logging.getLogger("comtypes").setLevel(logging.WARNING)
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return None
    try:
        names = FilterGraph().get_input_devices()
    except Exception:
        log.debug("DirectShow enumeration failed", exc_info=True)
        return None
    return [CameraDevice(index=i, name=name) for i, name in enumerate(names)]


def list_devices() -> Discovery:
    """List attached cameras without opening any of them."""
    devices = _enumerate_directshow()
    if devices is not None:
        return Discovery(devices=devices, method="directshow")
    return Discovery(
        devices=[CameraDevice(index=i, name=f"Camera {i}") for i in range(4)],
        method="assumed",
        warning=(
            "Could not enumerate cameras by name (pygrabber unavailable). "
            "Showing index guesses; check each one to see which are real."
        ),
    )


def check_available(index: int, *, backend: str | None = None,
                    width: int = 640, height: int = 480) -> CameraDevice:
    """Open one camera and report whether it actually delivers a frame.

    Opening is the only way to know, and it briefly lights the privacy LED.
    Kept out of :func:`list_devices` for exactly that reason.
    """
    from .capture import CameraOpenError, open_camera

    name = f"Camera {index}"
    for device in list_devices().devices:
        if device.index == index:
            name = device.name
            break
    try:
        cap, used = open_camera(index, width, height, backend)
    except CameraOpenError as exc:
        return CameraDevice(index, name, available=False, detail=exc.summary)
    size = (int(cap.get(3)) or 0, int(cap.get(4)) or 0)
    cap.release()
    return CameraDevice(index, name, available=True, backend=used, size=size)


def describe(index: int) -> str:
    """Best-effort human name for one index, for logs and error messages."""
    for device in list_devices().devices:
        if device.index == index:
            return f"{device.name} (index {index})"
    return f"camera index {index}"
