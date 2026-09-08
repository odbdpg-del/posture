"""Command line entry points.

    python -m posture                     control panel (default)
    python -m posture --list-cameras      what is attached
    python -m posture --diagnose          what one camera can actually see
    python -m posture --watch --debug     terminal metrics for one camera

The control panel is the normal way in. The terminal modes stay because they
are the fastest way to check a metric or a camera without a browser.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from . import metrics as met
from . import overlay
from .capture import CameraCapture, CameraOpenError
from .config import Config
from .pose import PoseEstimator

log = logging.getLogger("posture")


def list_cameras(check: bool = True) -> int:
    """Print attached cameras by name, optionally opening each to test it."""
    from .devices import check_available, list_devices

    found = list_devices()
    if found.warning:
        print(f"note: {found.warning}\n")
    if not found.devices:
        print("No cameras found.")
        return 1

    print(f"{len(found.devices)} device(s), listed via {found.method}:\n")
    for device in found.devices:
        if check:
            # Enumeration is free; opening is not, so it only happens here and
            # only because the point of this command is to say what works.
            device = check_available(device.index)
        bits = [f"index {device.index}", device.name]
        if device.virtual:
            bits.append("virtual camera")
        if device.available is True:
            bits.append(f"working via {device.backend}")
            if device.size:
                bits.append(f"{device.size[0]}x{device.size[1]}")
        elif device.available is False:
            bits.append("NOT RESPONDING")
        print("  " + "   |  ".join(bits))
        if device.available is False and device.detail:
            print(f"      {device.detail}")
    return 0


def diagnose(index: int, cfg: Config, model_path: str | None, seconds: float = 12.0) -> int:
    """Sample for a few seconds and report what this camera can actually see.

    Camera placement is the thing most likely to be wrong, and it fails
    quietly: the model happily reports a pose from a head-and-shoulders crop,
    but the side metrics need hips that are not in the picture. Rather than
    leave that to be discovered as a permanently blank readout, measure
    per-landmark visibility and say which role the camera can support.
    """
    from . import landmarks as lmk

    try:
        estimator = PoseEstimator(model_path)
        capture = CameraCapture(index, sample_hz=cfg.sampling.fps).start()
    except (FileNotFoundError, CameraOpenError) as exc:
        print(exc, file=sys.stderr)
        return 2

    thresh = cfg.sampling.visibility_threshold
    seen: dict[str, list[float]] = {name: [] for name in lmk.NAMES.values()}
    usable = {"side": 0, "front": 0}
    frames = people = 0

    last_shown = -1
    print(f"Watching camera {index} for {seconds:g}s. Sit the way you normally work.")
    try:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            frame = capture.read(timeout=2.0)
            if frame is None:
                continue
            frames += 1
            arr = estimator.detect(frame.image, int(frame.t * 1000)).array
            if arr is None:
                continue
            people += 1
            for i, name in lmk.NAMES.items():
                seen[name].append(float(arr[i, 3]))
            for role in ("side", "front"):
                if met.compute(role, arr, frame.aspect, thresh, frame.t).usable:
                    usable[role] += 1
            # Refresh once a second rather than once a frame, so the countdown
            # stays readable when stdout is a pipe and \r does nothing.
            left = end - time.monotonic()
            if int(left) != last_shown:
                last_shown = int(left)
                sys.stdout.write(f"\r  {left:4.1f}s left, {people}/{frames} frames "
                                 "with a person ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        estimator.close()

    print("\n")
    if not people:
        print("No person detected at all. Check that you are in frame and lit from "
              "the front rather than backlit by a window.")
        return 1

    print(f"{people} of {frames} frames contained a person.\n")
    print(f"  {'landmark':<16}{'mean vis':>9}{'usable':>9}")
    for name, values in seen.items():
        if not values:
            continue
        ok = 100.0 * sum(v >= thresh for v in values) / len(values)
        flag = "" if ok >= 80 else ("  <- never seen" if ok == 0 else "  <- intermittent")
        print(f"  {name:<16}{sum(values) / len(values):9.2f}{ok:8.0f}%{flag}")

    print()
    for role in ("side", "front"):
        pct = 100.0 * usable[role] / people
        verdict = "good" if pct >= 80 else ("marginal" if pct >= 30 else "not viable")
        print(f"  {role:<6} role: metrics available in {pct:5.1f}% of frames  ({verdict})")

    hip_ok = any(any(v >= thresh for v in values)
                 for name, values in seen.items() if name.endswith("hip"))
    print()
    if not hip_ok:
        print("Your hips are never in frame, so the side metrics (neck flexion,\n"
              "forward head, torso lean) cannot be computed at all. A side camera\n"
              "needs to see you from ear to hip: move it further back, lower it to\n"
              "roughly chest height, and place it level with your shoulder to one\n"
              "side rather than in front of you.")
    elif usable["front"] > usable["side"]:
        print("This camera is better suited to the 'front' role than 'side'.")
    return 0


def run_ui(args: argparse.Namespace) -> int:
    """Start the cameras and the local control panel, and wait.

    This is the default mode. It exits on Ctrl-C or when the panel's stop
    button is pressed, and it always tears the cameras down on the way out --
    an app that watches you through a webcam has no business being hard to
    stop.
    """
    import webbrowser

    from .monitor import Monitor
    from .tray import TrayIcon
    from .webui import AlreadyRunning, serve

    cfg = Config.load(args.config)
    if args.fps:
        cfg.sampling.fps = args.fps
    if args.port:
        cfg.web.port = args.port

    monitor = Monitor(cfg)
    # Claim the port before touching the cameras. Starting the monitor
    # first meant a second copy spent fifteen seconds opening webcams --
    # taking them from the copy already running -- only to find the port
    # gone and exit.
    try:
        server, panel = serve(monitor, cfg.web,
                              Path(args.config) if args.config else None)
    except AlreadyRunning as exc:
        print(exc, file=sys.stderr)
        return 2
    except OSError as exc:
        print("Could not start the control panel on port "
              f"{cfg.web.port}: {exc}", file=sys.stderr)
        return 2
    monitor.start()

    url = f"http://{cfg.web.host}:{cfg.web.port}/"

    tray = None
    if cfg.web.tray_icon:
        tray = TrayIcon(
            on_open=lambda: webbrowser.open(url),
            on_snooze=monitor.snooze,
            on_resume=monitor.cancel_snooze,
            on_calibrate=lambda: monitor.start_calibration(None),
            on_quit=panel.shutdown_requested.set,
        )
        if tray.start():
            log.info("tray icon started")
        else:
            log.warning("tray icon unavailable; the subtle alert step will only "
                        "show in the panel")
            tray = None

    print(f"Posture is running. Control panel: {url}")
    print("Press Ctrl-C here"
          + (", use the tray icon" if tray else "")
          + ", or use 'Stop monitoring' in the panel, to quit.")
    if cfg.web.open_browser and not args.no_browser:
        webbrowser.open(url)

    try:
        while not panel.shutdown_requested.wait(0.5):
            # The tray follows the same snapshot the panel reads, so the two
            # can never disagree about what colour the state is.
            if tray is not None:
                tray.update(monitor.snapshot())
    except KeyboardInterrupt:
        print()
    finally:
        print("stopping...")
        if tray is not None:
            tray.stop()
        server.shutdown()
        server.server_close()
        monitor.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m posture",
        description="Local posture monitor. With no arguments, opens the control panel.",
    )
    p.add_argument("--camera", type=int, default=None, metavar="INDEX",
                   help="camera index (default: first camera in config)")
    p.add_argument("--role", choices=("side", "front"), default=None,
                   help="what this camera can see (default: from config)")
    p.add_argument("--fps", type=float, default=None,
                   help="sampling rate in Hz (default: from config, 5)")
    p.add_argument("--vis", type=float, default=None, metavar="T",
                   help="landmark visibility threshold 0..1 (default: from config)")
    p.add_argument("--model", default=None, help="path to the .task pose model")
    p.add_argument("--config", default=None, help="path to config.json")
    p.add_argument("--debug", action="store_true",
                   help="show a window with the skeleton overlay")
    p.add_argument("--verbose", action="store_true",
                   help="print one line per sample instead of a refreshing line")
    p.add_argument("--duration", type=float, default=None, metavar="SEC",
                   help="stop after this many seconds (default: run until Ctrl-C)")
    p.add_argument("--list-cameras", action="store_true",
                   help="list attached cameras and exit")
    p.add_argument("--diagnose", action="store_true",
                   help="check what this camera can see and which role it suits")
    p.add_argument("--watch", action="store_true",
                   help="terminal metric view for one camera instead of the panel")
    p.add_argument("--port", type=int, default=None,
                   help="port for the control panel (default: from config, 8760)")
    p.add_argument("--no-browser", action="store_true",
                   help="do not open a browser when the panel starts")
    return p


def format_line(sample: met.MetricSample, status: str, fps: float, infer_ms: float,
                cpu_pct: float, verdict=None) -> str:
    parts = [f"{status:<24}"]
    for spec in met.SPECS_BY_ROLE[sample.role]:
        value = sample.values.get(spec.key)
        parts.append(f"{spec.short}={'  --  ' if value is None else f'{value:+7.2f}'}")
    if verdict is not None:
        # The worst metric as a fraction of its own tolerance: one number that
        # says how close to an alert you are, independent of units.
        parts.append(f"| {verdict.state.upper():<7} {verdict.worst_ratio:4.2f}x")
    parts.append(f"| {fps:4.1f}fps {infer_ms:5.1f}ms {cpu_pct:4.1f}%cpu")
    if sample.missing:
        parts.append(f"| faint: {','.join(sample.missing)}")
    return "  ".join(parts)


def run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    cam_cfg = cfg.cameras[0]
    index = args.camera if args.camera is not None else cam_cfg.index
    role = args.role or cam_cfg.role
    fps = args.fps or cfg.sampling.fps
    vis_thresh = args.vis if args.vis is not None else cfg.sampling.visibility_threshold
    model_path = args.model or cfg.model_path

    try:
        estimator = PoseEstimator(
            model_path,
            min_detection_confidence=cfg.sampling.min_detection_confidence,
            min_presence_confidence=cfg.sampling.min_presence_confidence,
            min_tracking_confidence=cfg.sampling.min_tracking_confidence,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        capture = CameraCapture(index, sample_hz=fps, width=cam_cfg.width,
                                height=cam_cfg.height, backend=cam_cfg.backend)
        capture.start()
    except CameraOpenError as exc:
        print(exc, file=sys.stderr)
        estimator.close()
        return 2

    print(f"camera {index} via {capture.backend_name}, role={role}, "
          f"{fps:g} fps, visibility>={vis_thresh:g}")
    for spec in met.SPECS_BY_ROLE[role]:
        print(f"  {spec.key:<15} [{spec.unit}]  {spec.description}")
    print("Ctrl-C to stop." + ("  (q or Esc in the window also works)" if args.debug else ""))
    print()

    # Terminal mode runs the detector too, so a threshold change can be checked
    # here without a browser. It calibrates from the first few seconds of the
    # run unless the config already holds a baseline for this camera.
    from . import detector as detmod
    from .calibration import CalibrationSession
    from .monitor import _baselines, _settings

    detector = detmod.PostureDetector(_settings(cfg), _baselines(cfg))
    calibrating: CalibrationSession | None = None
    if not detector.calibrated:
        calibrating = CalibrationSession(index, role,
                                         duration=cfg.calibration.duration,
                                         min_samples=cfg.calibration.min_samples)
        calibrating.start()
        print(f"No baseline for this camera. Calibrating for "
              f"{cfg.calibration.duration:g}s -- sit the way you want to sit.")

    window = f"posture debug - {role} - camera {index}"
    started = time.monotonic()
    # Duty cycle measured from process CPU time over wall time. This is the
    # number the <10%-of-a-core budget is about, and process_time() sums every
    # thread, so the capture thread is included rather than hidden.
    cpu0, wall0 = time.process_time(), time.monotonic()
    cpu_pct = 0.0
    infer_avg = 0.0
    seen = 0
    interval_start = time.monotonic()
    interval_frames = 0
    measured_fps = 0.0
    exit_code = 0

    try:
        while True:
            frame = capture.read(timeout=2.0)
            now = time.monotonic()

            if frame is None:
                msg = capture.last_error or "no frames from camera"
                sys.stdout.write(f"\r{'CAMERA STALLED: ' + msg:<100}")
                sys.stdout.flush()
                if args.duration and now - started >= args.duration:
                    break
                continue

            result = estimator.detect(frame.image, int(frame.t * 1000))
            sample = met.compute(role, result.array, frame.aspect, vis_thresh,
                                 frame.t, camera_index=index)
            status, colour = overlay.status_for(sample)

            if calibrating is not None:
                calibrating.add(sample)
                if not calibrating.running:
                    baseline, problems = calibrating.result()
                    calibrating = None
                    detector.set_baselines({index: baseline})
                    summary = ", ".join(
                        f"{k}={v.centre:.2f}+-{v.spread:.2f}"
                        for k, v in baseline.metrics.items())
                    print()
                    print("Baseline: " + (summary or "nothing usable"))
                    for problem in problems:
                        print(f"  ! {problem}")
            verdict = detector.update([sample], now=frame.t)

            seen += 1
            interval_frames += 1
            infer_avg += (result.infer_ms - infer_avg) / min(seen, 30)
            if now - interval_start >= 1.0:
                measured_fps = interval_frames / (now - interval_start)
                interval_start, interval_frames = now, 0
                cpu1, wall1 = time.process_time(), time.monotonic()
                if wall1 > wall0:
                    cpu_pct = 100.0 * (cpu1 - cpu0) / (wall1 - wall0)
                cpu0, wall0 = cpu1, wall1

            line = format_line(sample, status, measured_fps, infer_avg, cpu_pct,
                               verdict)
            if args.verbose:
                print(f"{frame.t - started:7.2f}s  {line}")
            else:
                sys.stdout.write(f"\r{line:<160}")
                sys.stdout.flush()

            if args.debug:
                canvas = frame.image.copy()
                overlay.draw_skeleton(canvas, result.array, role, vis_thresh)
                overlay.draw_panel(canvas, sample, status=status, status_colour=colour,
                                   fps=measured_fps, infer_ms=infer_avg, cpu_pct=cpu_pct)
                cv2.imshow(window, canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                # Clicking the window's X destroys it but does not tell us, and
                # the next imshow silently builds a new one -- so the window
                # reappears every frame and looks like an app that refuses to
                # close. Treat "no longer visible" as a quit request.
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.duration and now - started >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        estimator.close()
        if args.debug:
            cv2.destroyAllWindows()
        print()
        elapsed = time.monotonic() - started
        print(f"stopped after {elapsed:.1f}s, {seen} samples "
              f"({seen / elapsed if elapsed else 0:.2f}/s), "
              f"mean inference {infer_avg:.1f} ms, ~{cpu_pct:.1f}% of one core")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.list_cameras:
        return list_cameras()

    if args.diagnose:
        cfg = Config.load(args.config)
        index = args.camera if args.camera is not None else cfg.cameras[0].index
        return diagnose(index, cfg, args.model or cfg.model_path)

    if args.watch or args.debug:
        return run(args)

    return run_ui(args)


if __name__ == "__main__":
    raise SystemExit(main())
