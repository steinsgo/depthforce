"""Validation B: render a static GPU particle array with OpenGLRenderer."""

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from warp_env import configure_warp_environment, configure_warp_native_path

configure_warp_environment(ROOT)

import warp as wp

configure_warp_native_path(wp)

import warp.render

from config import AppConfig
from rendering import configure_particle_geometry
from simulation import ParticleSystem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--particles", type=int, default=150_000)
    args = parser.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("FAIL: CUDA is unavailable.", file=sys.stderr)
        return 1
    config = AppConfig()
    config.simulation.particle_count = args.particles
    particles = ParticleSystem(config.simulation)
    render = config.render
    renderer = wp.render.OpenGLRenderer(
        title="DepthForce renderer test",
        fps=render.target_fps,
        screen_width=render.width,
        screen_height=render.height,
        near_plane=0.05,
        far_plane=50.0,
        camera_fov=render.camera_fov,
        camera_pos=render.camera_pos,
        camera_front=render.camera_front,
        background_color=render.background_color,
        draw_grid=False,
        draw_sky=False,
        draw_axis=False,
        show_info=False,
        vsync=False,
        device=particles.device,
    )
    configure_particle_geometry(renderer)

    start = time.perf_counter()
    frames = 0
    try:
        while renderer.is_running() and frames < args.frames:
            renderer.begin_frame(frames / render.target_fps)
            renderer.render_points(
                "particles",
                particles.initial_positions if frames == 0 else particles.positions,
                radius=config.simulation.point_radius,
                colors=particles.colors if frames == 0 else None,
                as_spheres=False,
            )
            renderer.end_frame()
            frames += 1
    finally:
        elapsed = max(time.perf_counter() - start, 1.0e-6)
        if renderer.is_running():
            renderer.close()
    print(
        f"PASS: rendered {args.particles:,} GPU particles for {frames} frames "
        f"at {frames / elapsed:.1f} average FPS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
