"""Validation A: compile and execute a tiny Warp CUDA kernel."""

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from warp_env import configure_warp_environment, configure_warp_native_path

configure_warp_environment(ROOT)

import numpy as np
import warp as wp

configure_warp_native_path(wp)


@wp.kernel
def saxpy(values: wp.array(dtype=float), multiplier: float, offset: float):
    index = wp.tid()
    values[index] = values[index] * multiplier + offset


def main() -> int:
    wp.init()
    print(f"Warp {wp.__version__}")
    print(f"Devices: {wp.get_devices()}")
    if not wp.is_cuda_available():
        print("FAIL: Warp did not detect a CUDA device.", file=sys.stderr)
        return 1

    device = wp.get_device("cuda:0")
    count = 4096
    values = wp.array(np.arange(count, dtype=np.float32), dtype=float, device=device)
    start = time.perf_counter()
    wp.launch(saxpy, dim=count, inputs=[values, 2.0, 1.0], device=device)
    wp.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result = values.numpy()
    expected = np.arange(count, dtype=np.float32) * 2.0 + 1.0
    if not np.array_equal(result, expected):
        print("FAIL: CUDA kernel result did not match the expected values.", file=sys.stderr)
        return 1
    print(f"PASS: CUDA kernel compiled and ran correctly on {device} ({elapsed_ms:.2f} ms including warm-up).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
