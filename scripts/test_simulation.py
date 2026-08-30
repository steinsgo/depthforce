"""Validation C: benchmark force interaction and verify visible displacement."""

import argparse
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

from config import AppConfig
from interaction import ForceSources
from simulation import ParticleSystem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=150_000)
    parser.add_argument("--sources", type=int, default=96)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    config = AppConfig()
    config.simulation.particle_count = args.particles
    source_count = min(max(args.sources, 1), config.simulation.max_force_sources)
    wp.init()
    particles = ParticleSystem(config.simulation)

    rng = np.random.default_rng(42)
    positions = rng.uniform(
        low=(-2.4, -1.4, -0.45),
        high=(2.4, 1.4, 0.45),
        size=(source_count, 3),
    ).astype(np.float32)
    directions = rng.normal(size=(source_count, 3)).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-6)
    sources = ForceSources(
        positions=positions,
        directions=directions,
        strengths=np.full(source_count, config.simulation.force_strength, dtype=np.float32),
        radii=np.full(source_count, config.simulation.force_radius, dtype=np.float32),
    )
    particles.set_force_sources(sources)

    # Compile/warm up independently of the timed section.
    particles.step(1.0 / 60.0, 0.0)
    wp.synchronize_device(particles.device)
    start = time.perf_counter()
    for step in range(args.steps):
        particles.step(1.0 / 60.0, (step + 1) / 60.0)
    wp.synchronize_device(particles.device)
    elapsed = max(time.perf_counter() - start, 1.0e-6)

    final_positions = particles.positions.numpy()
    displacement = np.linalg.norm(final_positions - particles.initial_positions, axis=1)
    maximum = float(displacement.max())
    percentile_95 = float(np.percentile(displacement, 95.0))
    if maximum < 0.05:
        print("FAIL: force sources did not create a visible particle displacement.", file=sys.stderr)
        return 1
    print(
        f"PASS: {args.particles:,} particles x {source_count} sources ran at "
        f"{args.steps / elapsed:.1f} simulation steps/s; displacement p95={percentile_95:.3f}, "
        f"max={maximum:.3f} scene units."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
