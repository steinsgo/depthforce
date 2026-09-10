"""Validation D: enumerate a RealSense device and stream metric depth."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from camera import RealSenseCamera, RealSenseUnavailableError, list_realsense_devices
from config import AppConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    try:
        config = AppConfig().camera
        devices = list_realsense_devices()
        if not devices:
            print("NO DEVICE: the native Windows RealSense SDK enumerated zero devices.")
            print("Connect the D435i to a USB 3 port, then rerun this test.")
            return 2
        for device in devices:
            print(f"Found: {device.name} | serial={device.serial} | firmware={device.firmware}")

        camera = RealSenseCamera(
            width=config.width,
            height=config.height,
            fps=config.fps,
            visual_preset=config.visual_preset,
            spatial_filter_enabled=config.spatial_filter_enabled,
            spatial_filter_magnitude=config.spatial_filter_magnitude,
            spatial_filter_alpha=config.spatial_filter_alpha,
            spatial_filter_delta=config.spatial_filter_delta,
        )
        camera.start()
        assert camera.intrinsics is not None
        print(
            f"Profile: {config.width}x{config.height}@{config.fps} | "
            f"preset={camera.active_visual_preset} | spatial_filter=light"
        )
        if camera.active_visual_preset.casefold() != config.visual_preset.casefold():
            print(
                f"FAIL: requested preset={config.visual_preset}, "
                f"camera reports {camera.active_visual_preset}",
                file=sys.stderr,
            )
            camera.stop()
            return 1
        print(f"Depth scale: {camera.depth_scale:.8f} m/unit")
        print(f"Intrinsics: {camera.intrinsics}")
        valid_ratios = []
        device_timestamps = []
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
                device_timestamps.append(frame.timestamp_ms)
                captured += 1
        finally:
            camera.stop()
        elapsed = max(time.perf_counter() - start, 1.0e-6)
        device_elapsed = max((device_timestamps[-1] - device_timestamps[0]) * 0.001, 1.0e-6)
        device_fps = (captured - 1) / device_elapsed
        print(
            f"PASS: captured {captured} depth frames at host={captured / elapsed:.1f} FPS, "
            f"device={device_fps:.1f} FPS; "
            f"mean valid depth={np.mean(valid_ratios) * 100.0:.1f}%."
        )
        return 0
    except RealSenseUnavailableError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
