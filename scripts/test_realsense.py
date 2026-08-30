"""Validation D: enumerate a RealSense device and stream metric depth."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from camera import RealSenseCamera, RealSenseUnavailableError, list_realsense_devices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    try:
        devices = list_realsense_devices()
        if not devices:
            print("NO DEVICE: the native Windows RealSense SDK enumerated zero devices.")
            print("Connect the D435i to a USB 3 port, then rerun this test.")
            return 2
        for device in devices:
            print(f"Found: {device.name} | serial={device.serial} | firmware={device.firmware}")

        camera = RealSenseCamera(640, 480, 30)
        camera.start()
        assert camera.intrinsics is not None
        print(f"Depth scale: {camera.depth_scale:.8f} m/unit")
        print(f"Intrinsics: {camera.intrinsics}")
        valid_ratios = []
        captured = 0
        start = time.perf_counter()
        try:
            while captured < args.frames:
                frame = camera.wait(timeout_ms=2500)
                if frame is None:
                    print("WARN: timed out waiting for a depth frame.")
                    continue
                valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0.0)
                valid_ratios.append(float(valid.mean()))
                captured += 1
        finally:
            camera.stop()
        elapsed = max(time.perf_counter() - start, 1.0e-6)
        print(
            f"PASS: captured {captured} depth frames at {captured / elapsed:.1f} FPS; "
            f"mean valid depth={np.mean(valid_ratios) * 100.0:.1f}%."
        )
        return 0
    except RealSenseUnavailableError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
