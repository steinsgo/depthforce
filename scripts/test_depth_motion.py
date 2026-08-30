"""Validation E/F: exercise depth motion and 3D clustering without hardware."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from camera import DepthIntrinsics
from config import AppConfig
from interaction import DepthMotionExtractor


def main() -> int:
    config = AppConfig()
    extractor = DepthMotionExtractor(config.camera, config.motion, config.simulation)
    intrinsics = DepthIntrinsics(width=640, height=480, fx=385.0, fy=385.0, ppx=320.0, ppy=240.0)

    background = np.full((480, 640), 1.25, dtype=np.float32)
    first = extractor.process(background, intrinsics, config.simulation.force_strength)
    if len(first.sources) != 0:
        print("FAIL: the first frame generated motion sources.", file=sys.stderr)
        return 1

    moved = background.copy()
    moved[155:325, 225:415] = 0.82
    second = extractor.process(moved, intrinsics, config.simulation.force_strength)
    if second.active_pixels == 0 or len(second.sources) == 0:
        print("FAIL: a large approaching depth patch generated no sources.", file=sys.stderr)
        return 1
    if len(second.sources) > config.simulation.max_force_sources:
        print("FAIL: source count exceeded the configured maximum.", file=sys.stderr)
        return 1
    if not np.all(np.isfinite(second.sources.positions)):
        print("FAIL: source positions contain invalid values.", file=sys.stderr)
        return 1
    print(
        f"PASS: {second.active_pixels} active low-resolution pixels became "
        f"{len(second.sources)} finite 3D force sources; max delta={second.maximum_delta:.3f} m."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
