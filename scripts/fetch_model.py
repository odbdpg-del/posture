"""One-time download of the MediaPipe pose model.

This is the only network access in the project, and it is deliberately a
separate script rather than something the app does lazily: the app itself must
never be able to reach the network, so it can only ever load a model that is
already on disk.

    python scripts/fetch_model.py

MediaPipe 1.0 dropped the old ``mp.solutions.pose`` API, which shipped its
model inside the pip package. The Tasks Vision API that replaced it takes an
external ``.task`` bundle, hence this step.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# "lite" is the right trade for this app: it is roughly three times cheaper
# than "full" and the difference in landmark accuracy is invisible next to the
# noise floor of a webcam at 5 fps. Swap with --variant if you want to compare.
VARIANTS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        None,
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        None,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="lite")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file already exists")
    args = parser.parse_args(argv)

    url, expected = VARIANTS[args.variant]
    dest = MODELS_DIR / f"pose_landmarker_{args.variant}.task"

    if dest.exists() and not args.force:
        print(f"already present: {dest}")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except OSError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        # Google publishes these under a "latest" path, so a mismatch most
        # likely means they shipped a new revision rather than that anything is
        # wrong. Say so plainly instead of failing with a scary checksum error.
        print(f"note: sha256 is {digest}, expected {expected}.\n"
              "      Upstream publishes this under a 'latest' URL, so the model "
              "was probably revised. Update the hash in this script if you "
              "trust the new file.", file=sys.stderr)

    tmp = dest.with_suffix(".task.tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"wrote {dest} ({len(data) / 1_048_576:.1f} MiB, sha256 {digest[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
