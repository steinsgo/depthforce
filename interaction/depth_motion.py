"""Turn low-resolution metric depth changes into a few 3D force sources."""

from dataclasses import dataclass

import cv2
import numpy as np

from camera.realsense import DepthIntrinsics
from config import CameraConfig, MotionConfig, SimulationConfig


@dataclass(slots=True)
class ForceSources:
    positions: np.ndarray
    directions: np.ndarray
    strengths: np.ndarray
    radii: np.ndarray

    @classmethod
    def empty(cls) -> "ForceSources":
        return cls(
            positions=np.empty((0, 3), dtype=np.float32),
            directions=np.empty((0, 3), dtype=np.float32),
            strengths=np.empty((0,), dtype=np.float32),
            radii=np.empty((0,), dtype=np.float32),
        )

    def __len__(self) -> int:
        return int(self.strengths.shape[0])


@dataclass(slots=True)
class MotionResult:
    sources: ForceSources
    depth_small: np.ndarray
    motion_mask: np.ndarray
    active_pixels: int
    average_delta: float
    maximum_delta: float


class DepthMotionExtractor:
    def __init__(
        self,
        camera: CameraConfig,
        motion: MotionConfig,
        simulation: SimulationConfig,
    ):
        self.camera = camera
        self.motion = motion
        self.simulation = simulation
        self._previous_depth: np.ndarray | None = None
        self._kernel = np.ones((3, 3), dtype=np.uint8)

    def reset(self) -> None:
        self._previous_depth = None

    def process(
        self,
        depth_m: np.ndarray,
        intrinsics: DepthIntrinsics,
        force_strength: float,
    ) -> MotionResult:
        if depth_m.ndim != 2:
            raise ValueError(f"Expected a 2D depth frame, got shape {depth_m.shape}.")

        small = cv2.resize(
            depth_m,
            (self.motion.downsample_width, self.motion.downsample_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32, copy=False)
        valid_current = (
            np.isfinite(small)
            & (small >= self.camera.depth_min)
            & (small <= self.camera.depth_max)
        )
        clean = np.where(valid_current, small, 0.0).astype(np.float32, copy=False)

        if self._previous_depth is None:
            self._previous_depth = clean.copy()
            return MotionResult(
                sources=ForceSources.empty(),
                depth_small=clean,
                motion_mask=np.zeros(clean.shape, dtype=np.uint8),
                active_pixels=0,
                average_delta=0.0,
                maximum_delta=0.0,
            )

        previous = self._previous_depth
        valid_previous = previous > 0.0
        common_valid = valid_current & valid_previous

        smoothed = clean.copy()
        alpha = self.motion.depth_ema_alpha
        smoothed[common_valid] = (
            alpha * clean[common_valid] + (1.0 - alpha) * previous[common_valid]
        )

        # Positive signed delta means the surface moved toward the camera.
        signed_delta = np.where(common_valid, previous - smoothed, 0.0)
        magnitude = np.abs(signed_delta)
        raw_mask = (
            common_valid
            & (magnitude >= self.motion.motion_threshold)
        )

        mask_u8 = raw_mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, self._kernel)
        mask_u8 = cv2.dilate(mask_u8, self._kernel, iterations=1)
        active_mask = mask_u8 > 0

        sources = self._cluster_sources(
            smoothed,
            signed_delta,
            magnitude,
            active_mask,
            intrinsics,
            force_strength,
        )
        active_magnitude = magnitude[active_mask]
        average_delta = float(active_magnitude.mean()) if active_magnitude.size else 0.0
        maximum_delta = float(active_magnitude.max()) if active_magnitude.size else 0.0

        self._previous_depth = smoothed
        return MotionResult(
            sources=sources,
            depth_small=smoothed,
            motion_mask=mask_u8,
            active_pixels=int(active_mask.sum()),
            average_delta=average_delta,
            maximum_delta=maximum_delta,
        )

    def _cluster_sources(
        self,
        depth: np.ndarray,
        signed_delta: np.ndarray,
        magnitude: np.ndarray,
        active_mask: np.ndarray,
        intrinsics: DepthIntrinsics,
        force_strength: float,
    ) -> ForceSources:
        candidates: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
        height, width = depth.shape
        cell_h = self.motion.cell_height
        cell_w = self.motion.cell_width

        for y0 in range(0, height, cell_h):
            y1 = min(y0 + cell_h, height)
            for x0 in range(0, width, cell_w):
                x1 = min(x0 + cell_w, width)
                cell_mask = active_mask[y0:y1, x0:x1]
                active_count = int(cell_mask.sum())
                if active_count < self.motion.min_active_pixels_per_cell:
                    continue

                local_y, local_x = np.nonzero(cell_mask)
                global_y = local_y + y0
                global_x = local_x + x0
                weights = magnitude[global_y, global_x]
                weight_sum = float(weights.sum())
                if weight_sum <= 1.0e-8:
                    continue

                center_x = float(np.dot(global_x, weights) / weight_sum)
                center_y = float(np.dot(global_y, weights) / weight_sum)
                source_depth = float(np.dot(depth[global_y, global_x], weights) / weight_sum)
                mean_signed = float(np.dot(signed_delta[global_y, global_x], weights) / weight_sum)
                mean_magnitude = float(weights.mean())

                full_u = (center_x + 0.5) * intrinsics.width / width - 0.5
                full_v = (center_y + 0.5) * intrinsics.height / height - 0.5
                camera_x = (full_u - intrinsics.ppx) / intrinsics.fx * source_depth
                camera_y = (full_v - intrinsics.ppy) / intrinsics.fy * source_depth
                position = np.array(
                    [
                        camera_x * self.motion.scene_xy_scale,
                        -camera_y * self.motion.scene_xy_scale,
                        (self.motion.scene_depth_center - source_depth) * self.motion.scene_z_scale,
                    ],
                    dtype=np.float32,
                )

                direction_sign = 1.0 if mean_signed >= 0.0 else -1.0
                direction = np.array([0.0, 0.0, direction_sign], dtype=np.float32)
                normalized_motion = np.clip(
                    (mean_magnitude - self.motion.motion_threshold)
                    / max(self.motion.max_motion_delta - self.motion.motion_threshold, 1.0e-6),
                    0.0,
                    1.0,
                )
                approach = max(mean_signed, 0.0) / max(mean_magnitude, 1.0e-6)
                strength = force_strength * (0.38 + 2.25 * normalized_motion)
                strength *= 1.0 + self.motion.approach_boost * approach
                coverage = active_count / float(cell_h * cell_w)
                radius = self.simulation.force_radius * (0.78 + 0.55 * np.sqrt(coverage))
                score = strength * (0.5 + coverage)
                candidates.append((score, position, direction, float(strength), float(radius)))

        if not candidates:
            return ForceSources.empty()

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[: self.simulation.max_force_sources]
        return ForceSources(
            positions=np.stack([item[1] for item in candidates]).astype(np.float32),
            directions=np.stack([item[2] for item in candidates]).astype(np.float32),
            strengths=np.asarray([item[3] for item in candidates], dtype=np.float32),
            radii=np.asarray([item[4] for item in candidates], dtype=np.float32),
        )

    def make_debug_image(self, result: MotionResult) -> np.ndarray:
        depth = result.depth_small
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        valid = depth > 0.0
        if valid.any():
            clipped = np.clip(depth, self.camera.depth_min, self.camera.depth_max)
            normalized[valid] = (
                255.0
                * (self.camera.depth_max - clipped[valid])
                / (self.camera.depth_max - self.camera.depth_min)
            ).astype(np.uint8)
        depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        mask_color = np.zeros_like(depth_color)
        mask_color[:, :, 2] = result.motion_mask
        combined = cv2.addWeighted(depth_color, 0.68, mask_color, 0.75, 0.0)
        return cv2.resize(combined, (640, 480), interpolation=cv2.INTER_NEAREST)
