"""DepthForce real-time application entry point."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
CAMERA_DEBUG_WINDOW = "DepthForce controls + depth motion"
FOREGROUND_TRACKBAR = "Foreground gap mm"
PRESET_TRACKBAR = "Preset 0=Balanced 1=Punchy"
from warp_env import configure_warp_environment, configure_warp_native_path

configure_warp_environment(PROJECT_ROOT)

import cv2
import numpy as np
import warp as wp

configure_warp_native_path(wp)

from camera import RealSenseCamera, RealSenseUnavailableError
from config import AppConfig, INTERACTION_PRESETS, apply_interaction_preset
from interaction import DepthMotionExtractor, ForceSources, MotionResult
from rendering import configure_particle_geometry
from simulation import ParticleSystem


def synthetic_force_sources(
    simulation_time: float,
    force_strength: float,
    force_radius: float,
) -> ForceSources:
    position = np.array(
        [
            2.05 * math.sin(simulation_time * 0.73),
            0.95 * math.sin(simulation_time * 1.07 + 0.4),
            0.42 * math.cos(simulation_time * 0.61),
        ],
        dtype=np.float32,
    )
    velocity = np.array(
        [
            2.05 * 0.73 * math.cos(simulation_time * 0.73),
            0.95 * 1.07 * math.cos(simulation_time * 1.07 + 0.4),
            -0.42 * 0.61 * math.sin(simulation_time * 0.61),
        ],
        dtype=np.float32,
    )
    velocity /= max(float(np.linalg.norm(velocity)), 1.0e-6)
    return ForceSources(
        positions=position.reshape(1, 3),
        directions=velocity.reshape(1, 3),
        strengths=np.asarray([force_strength * 1.12], dtype=np.float32),
        radii=np.asarray([force_radius * 1.08], dtype=np.float32),
    )


class DepthForceApp:
    def __init__(self, args: argparse.Namespace, config: AppConfig):
        self.args = args
        self.config = config
        self.synthetic = bool(args.synthetic)
        self.debug = bool(args.debug)
        self.camera_debug = bool(args.camera_debug)
        self.interaction_enabled = True
        self.force_strength = config.simulation.force_strength
        self.current_preset = args.preset
        self._preset_names = tuple(INTERACTION_PRESETS)
        self._preset_slider_value = self._preset_names.index(self.current_preset)
        self.reset_requested = False
        self.latest_motion: MotionResult | None = None
        self.camera: RealSenseCamera | None = None
        self.motion_extractor: DepthMotionExtractor | None = None
        self.renderer = None
        self._camera_controls_ready = False

        wp.init()
        if not wp.is_cuda_available():
            devices = ", ".join(str(device) for device in wp.get_devices())
            raise RuntimeError(
                "CUDA is unavailable to NVIDIA Warp. "
                f"Detected Warp devices: {devices or 'none'}. Check the NVIDIA driver installation."
            )
        self.particles = ParticleSystem(config.simulation, device=args.device)

        if not self.synthetic:
            self.camera = RealSenseCamera(
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
                visual_preset=config.camera.visual_preset,
                spatial_filter_enabled=config.camera.spatial_filter_enabled,
                spatial_filter_magnitude=config.camera.spatial_filter_magnitude,
                spatial_filter_alpha=config.camera.spatial_filter_alpha,
                spatial_filter_delta=config.camera.spatial_filter_delta,
            )
            self.camera.start()
            assert self.camera.intrinsics is not None
            self.motion_extractor = DepthMotionExtractor(
                config.camera,
                config.motion,
                config.simulation,
            )

        try:
            self._create_renderer()
        except Exception:
            if self.camera is not None:
                self.camera.stop()
            raise

    def _create_renderer(self) -> None:
        try:
            import warp.render

            render = self.config.render
            self.renderer = wp.render.OpenGLRenderer(
                title="DepthForce",
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
                show_info=self.debug,
                vsync=False,
                device=self.particles.device,
            )
            configure_particle_geometry(self.renderer)
            self.renderer.register_key_press_callback(self._on_key_press)
            self.renderer.register_input_processor(self._consume_control_keys)
            if render.maximized:
                self.renderer.window.maximize()

            # Prime renderer allocation with the existing CPU initialization data.
            # Every real-time frame after this consumes the GPU positions directly.
            self.renderer.begin_frame(0.0)
            self.renderer.render_points(
                name="particles",
                points=self.particles.initial_positions,
                radius=self.config.simulation.point_radius,
                colors=self.particles.colors,
                as_spheres=False,
            )
            self.renderer.end_frame()
        except Exception as exc:
            raise RuntimeError(
                "Warp OpenGLRenderer could not be initialized. Ensure pyglet is installed, "
                "the NVIDIA display GPU is active, and run from native Windows rather than WSL. "
                f"Original error: {exc}"
            ) from exc

    def _on_key_press(self, symbol: int, modifiers: int):
        import pyglet
        from pyglet.window import key

        if symbol == key.Q:
            self.renderer.close()
            return pyglet.event.EVENT_HANDLED
        if symbol == key.R:
            self.reset_requested = True
            return pyglet.event.EVENT_HANDLED
        if symbol == key.SPACE:
            self.interaction_enabled = not self.interaction_enabled
            if not self.interaction_enabled:
                self.particles.clear_force_sources()
            print(f"RealSense/synthetic interaction: {'ON' if self.interaction_enabled else 'OFF'}")
            return pyglet.event.EVENT_HANDLED
        if symbol == key.D:
            self.debug = not self.debug
            self.renderer.show_info = self.debug
            if not self.debug and self.camera_debug:
                try:
                    cv2.destroyWindow(CAMERA_DEBUG_WINDOW)
                except cv2.error:
                    pass
                self._camera_controls_ready = False
            print(f"Debug information: {'ON' if self.debug else 'OFF'}")
            return pyglet.event.EVENT_HANDLED
        if symbol == key.M:
            self.config.motion.mirror_x = not self.config.motion.mirror_x
            if self.motion_extractor is not None:
                self.motion_extractor.reset_tracks()
            self._sync_control_trackbar("Mirror 0/1", int(self.config.motion.mirror_x))
            print(f"Mirror interaction: {'ON' if self.config.motion.mirror_x else 'OFF'}")
            return pyglet.event.EVENT_HANDLED
        if symbol == key._1:
            self._apply_preset("balanced")
            return pyglet.event.EVENT_HANDLED
        if symbol == key._2:
            self._apply_preset("punchy")
            return pyglet.event.EVENT_HANDLED
        if symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            self.force_strength *= 1.12
            self._sync_control_trackbar("Force", round(self.force_strength))
            print(f"Force strength: {self.force_strength:.2f}")
            return pyglet.event.EVENT_HANDLED
        if symbol in (key.MINUS, key.NUM_SUBTRACT):
            self.force_strength = max(1.0, self.force_strength / 1.12)
            self._sync_control_trackbar("Force", round(self.force_strength))
            print(f"Force strength: {self.force_strength:.2f}")
            return pyglet.event.EVENT_HANDLED
        return None

    @staticmethod
    def _consume_control_keys(key_handler):
        import pyglet
        from pyglet.window import key

        if any(
            key_handler[symbol]
            for symbol in (
                key.Q,
                key.R,
                key.SPACE,
                key.D,
                key.M,
                key._1,
                key._2,
                key.PLUS,
                key.EQUAL,
                key.MINUS,
            )
        ):
            return pyglet.event.EVENT_HANDLED
        return None

    def _ensure_camera_controls(self) -> None:
        if self._camera_controls_ready:
            return
        cv2.namedWindow(CAMERA_DEBUG_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_DEBUG_WINDOW, 680, 620)
        noop = lambda _value: None
        cv2.createTrackbar(
            "Force",
            CAMERA_DEBUG_WINDOW,
            min(120, max(1, round(self.force_strength))),
            120,
            noop,
        )
        cv2.createTrackbar(
            "Radius x100",
            CAMERA_DEBUG_WINDOW,
            min(160, max(20, round(self.config.simulation.force_radius * 100.0))),
            160,
            noop,
        )
        cv2.createTrackbar(
            FOREGROUND_TRACKBAR,
            CAMERA_DEBUG_WINDOW,
            min(80, max(5, round(self.config.motion.motion_threshold * 1000.0))),
            80,
            noop,
        )
        cv2.createTrackbar(
            PRESET_TRACKBAR,
            CAMERA_DEBUG_WINDOW,
            self._preset_slider_value,
            len(self._preset_names) - 1,
            noop,
        )
        cv2.createTrackbar(
            "Mirror 0/1",
            CAMERA_DEBUG_WINDOW,
            int(self.config.motion.mirror_x),
            1,
            noop,
        )
        self._camera_controls_ready = True

    def _apply_preset(self, name: str) -> None:
        apply_interaction_preset(self.config, name)
        self.current_preset = name
        self.force_strength = self.config.simulation.force_strength
        self._preset_slider_value = self._preset_names.index(name)
        if self.motion_extractor is not None:
            self.motion_extractor.reset_tracks()
        self._sync_control_trackbar("Force", round(self.force_strength))
        self._sync_control_trackbar(
            "Radius x100", round(self.config.simulation.force_radius * 100.0)
        )
        self._sync_control_trackbar(
            FOREGROUND_TRACKBAR, round(self.config.motion.motion_threshold * 1000.0)
        )
        self._sync_control_trackbar(PRESET_TRACKBAR, self._preset_slider_value)
        print(f"Interaction preset: {name.upper()}")

    def _read_camera_controls(self) -> None:
        self._ensure_camera_controls()
        try:
            selected_preset = cv2.getTrackbarPos(PRESET_TRACKBAR, CAMERA_DEBUG_WINDOW)
            if selected_preset != self._preset_slider_value:
                self._apply_preset(self._preset_names[selected_preset])
            self.force_strength = float(max(1, cv2.getTrackbarPos("Force", CAMERA_DEBUG_WINDOW)))
            self.config.simulation.force_radius = max(
                0.20,
                cv2.getTrackbarPos("Radius x100", CAMERA_DEBUG_WINDOW) / 100.0,
            )
            self.config.motion.motion_threshold = max(
                0.005,
                cv2.getTrackbarPos(FOREGROUND_TRACKBAR, CAMERA_DEBUG_WINDOW) / 1000.0,
            )
            self.config.motion.mirror_x = bool(
                cv2.getTrackbarPos("Mirror 0/1", CAMERA_DEBUG_WINDOW)
            )
        except cv2.error:
            self._camera_controls_ready = False

    def _sync_control_trackbar(self, name: str, value: int) -> None:
        if not self._camera_controls_ready:
            return
        try:
            cv2.setTrackbarPos(name, CAMERA_DEBUG_WINDOW, int(value))
        except cv2.error:
            self._camera_controls_ready = False

    def _update_camera_interaction(self, now: float) -> bool:
        if self.camera is None or self.motion_extractor is None:
            return False
        frame = self.camera.poll()
        if frame is None:
            return False

        if self.debug and self.camera_debug:
            self._read_camera_controls()
        assert self.camera.intrinsics is not None
        self.latest_motion = self.motion_extractor.process(
            frame.depth_m,
            self.camera.intrinsics,
            self.force_strength,
            frame.timestamp_ms,
        )
        self.particles.set_force_sources(self.latest_motion.sources)
        if self.debug and self.camera_debug:
            debug_image = self.motion_extractor.make_debug_image(self.latest_motion)
            cv2.imshow(CAMERA_DEBUG_WINDOW, debug_image)
            key_code = cv2.waitKey(1) & 0xFF
            if key_code in (ord("q"), 27):
                self.renderer.close()
        return True

    def run(self) -> None:
        mode = "synthetic" if self.synthetic else "RealSense"
        camera_description = ""
        if self.camera is not None and self.camera.device_info is not None:
            camera_description = f" | camera={self.camera.device_info.name} ({self.camera.device_info.serial})"
        print(
            f"DepthForce started | mode={mode} | particles={self.config.simulation.particle_count:,} "
            f"| device={self.particles.device}{camera_description}"
        )
        print(
            "Controls: ESC/Q quit | R reset/background | SPACE interaction | D debug | "
            "M mirror | 1 Balanced | 2 Punchy | +/- force"
        )
        print(
            f"Tuning: preset={self.current_preset} | force={self.force_strength:.1f} "
            f"| radius={self.config.simulation.force_radius:.2f} "
            f"| foreground_gap={self.config.motion.motion_threshold * 1000.0:.0f}mm "
            f"| mirror={'ON' if self.config.motion.mirror_x else 'OFF'}"
        )

        start_time = time.perf_counter()
        last_frame_time = start_time
        last_camera_frame_time = start_time
        last_metrics_time = start_time
        simulation_time = 0.0
        frame_count = 0
        interval_frames = 0
        interval_camera_frames = 0
        total_camera_frames = 0
        stale_sources_cleared = False

        try:
            while self.renderer.is_running():
                now = time.perf_counter()
                frame_dt = min(max(now - last_frame_time, 1.0 / 240.0), 0.05)
                last_frame_time = now

                if self.reset_requested:
                    self.particles.reset()
                    if self.motion_extractor is not None:
                        self.motion_extractor.reset()
                    self.reset_requested = False

                if self.interaction_enabled:
                    if self.synthetic:
                        self.particles.set_force_sources(
                            synthetic_force_sources(
                                simulation_time,
                                self.force_strength,
                                self.config.simulation.force_radius,
                            )
                        )
                    else:
                        got_camera_frame = self._update_camera_interaction(now)
                        if got_camera_frame:
                            last_camera_frame_time = now
                            stale_sources_cleared = False
                            interval_camera_frames += 1
                            total_camera_frames += 1
                        elif now - last_camera_frame_time > 0.10 and not stale_sources_cleared:
                            self.particles.clear_force_sources()
                            stale_sources_cleared = True
                else:
                    self.particles.clear_force_sources()

                substeps = min(3, max(1, math.ceil(frame_dt / self.config.simulation.max_step)))
                step_dt = frame_dt / substeps
                for _ in range(substeps):
                    simulation_time += step_dt
                    self.particles.step(step_dt, simulation_time)

                self.renderer.begin_frame(simulation_time)
                self.renderer.render_points(
                    name="particles",
                    points=self.particles.positions,
                    radius=self.config.simulation.point_radius,
                    as_spheres=False,
                )
                self.renderer.end_frame()

                frame_count += 1
                interval_frames += 1
                if self.args.frames and frame_count >= self.args.frames:
                    break

                metrics_elapsed = now - last_metrics_time
                if metrics_elapsed >= 1.0:
                    fps = interval_frames / metrics_elapsed
                    camera_fps = interval_camera_frames / metrics_elapsed
                    source_count = self.particles.source_count
                    motion = self.latest_motion
                    motion_text = (
                        f" | active={motion.active_pixels} | delta(avg/max)="
                        f"{motion.average_delta:.3f}/{motion.maximum_delta:.3f}m"
                        if motion is not None
                        else ""
                    )
                    caption = (
                        f"DepthForce | {fps:5.1f} FPS | {self.config.simulation.particle_count:,} particles "
                        f"| {source_count} sources | force {self.force_strength:.1f} "
                        f"| radius {self.config.simulation.force_radius:.2f}"
                    )
                    self.renderer.window.set_caption(caption if self.debug else "DepthForce")
                    if self.debug:
                        print(
                            f"FPS={fps:5.1f} | camera={camera_fps:4.1f} | sources={source_count:2d} "
                            f"| preset={self.current_preset} | device={self.particles.device} "
                            f"| force={self.force_strength:.1f} "
                            f"| radius={self.config.simulation.force_radius:.2f} "
                            f"| foreground_gap={self.config.motion.motion_threshold * 1000.0:.0f}mm "
                            f"| mirror={'on' if self.config.motion.mirror_x else 'off'}{motion_text}"
                        )
                    interval_frames = 0
                    interval_camera_frames = 0
                    last_metrics_time = now
        finally:
            elapsed = max(time.perf_counter() - start_time, 1.0e-6)
            average_fps = frame_count / elapsed
            camera_fps = total_camera_frames / elapsed
            if self.camera is not None:
                self.camera.stop()
            if self.camera_debug:
                cv2.destroyAllWindows()
            if self.renderer is not None and self.renderer.is_running():
                self.renderer.close()
            print(
                f"DepthForce stopped | frames={frame_count} | elapsed={elapsed:.2f}s "
                f"| average_fps={average_fps:.1f} | camera_fps={camera_fps:.1f}"
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Depth motion pushes a dense NVIDIA Warp particle field.")
    parser.add_argument("--synthetic", action="store_true", help="Run without a RealSense camera.")
    parser.add_argument("--particles", type=int, default=150_000, help="Particle count.")
    parser.add_argument("--device", default="cuda:0", help="Warp CUDA device, normally cuda:0.")
    parser.add_argument(
        "--maximized",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start the particle window maximized (default: enabled).",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(INTERACTION_PRESETS),
        default="balanced",
        help="Interaction feel: balanced is smooth; punchy is faster and stronger.",
    )
    parser.add_argument("--force-strength", type=float, default=None, help="Override repulsion strength.")
    parser.add_argument("--force-radius", type=float, default=None, help="Override force radius in scene units.")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="Override foreground/background depth separation in metres.",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mirror horizontal camera interaction (default: enabled).",
    )
    parser.add_argument("--debug", action="store_true", help="Print live performance and motion metrics.")
    parser.add_argument(
        "--camera-debug",
        action="store_true",
        help="Show the depth/motion OpenCV window while debug mode is enabled.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Exit after this many rendered frames (0 keeps running).",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.particles <= 0:
        print("ERROR: --particles must be positive.", file=sys.stderr)
        return 2

    config = AppConfig()
    apply_interaction_preset(config, args.preset)
    config.simulation.particle_count = args.particles
    if args.maximized is not None:
        config.render.maximized = args.maximized
    if args.force_strength is not None:
        config.simulation.force_strength = args.force_strength
    if args.force_radius is not None:
        config.simulation.force_radius = args.force_radius
    if args.motion_threshold is not None:
        config.motion.motion_threshold = args.motion_threshold
    if args.mirror is not None:
        config.motion.mirror_x = args.mirror
    if config.simulation.force_strength <= 0.0 or config.simulation.force_radius <= 0.0:
        print("ERROR: force strength and radius must be positive.", file=sys.stderr)
        return 2
    if config.motion.motion_threshold <= 0.0:
        print("ERROR: motion threshold must be positive.", file=sys.stderr)
        return 2
    try:
        app = DepthForceApp(args, config)
        app.run()
        return 0
    except RealSenseUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Connect the D435i over USB 3, or run: python main.py --synthetic", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
