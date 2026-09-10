"""Persistent depth foreground tracking and 3D force-source generation."""

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
    track_count: int
    background_ready: bool


@dataclass(slots=True)
class _ForceTrack:
    position: np.ndarray
    observed_position: np.ndarray
    velocity: np.ndarray
    direction: np.ndarray
    strength: float
    radius: float
    missed_frames: int = 0


class DepthMotionExtractor:
    """Build a persistent foreground model and track local sources in metric 3D."""

    def __init__(
        self,
        camera: CameraConfig,
        motion: MotionConfig,
        simulation: SimulationConfig,
    ):
        self.camera = camera
        self.motion = motion
        self.simulation = simulation
        self._background_depth: np.ndarray | None = None
        self._background_frames = 0
        self._previous_timestamp_ms: float | None = None
        self._kernel = np.ones((3, 3), dtype=np.uint8)
        self._tracks: list[_ForceTrack] = []

    def reset(self) -> None:
        self._background_depth = None
        self._background_frames = 0
        self._previous_timestamp_ms = None
        self._tracks.clear()

    def reset_tracks(self) -> None:
        """Drop source history while preserving the learned empty scene."""
        self._tracks.clear()

    def process(
        self,
        depth_m: np.ndarray,
        intrinsics: DepthIntrinsics,
        force_strength: float,
        timestamp_ms: float | None = None,
    ) -> MotionResult:
        if depth_m.ndim != 2:
            raise ValueError(f"Expected a 2D depth frame, got shape {depth_m.shape}.")

        clean = self._downsample_depth(depth_m)
        dt = self._frame_dt(timestamp_ms)

        if self._background_depth is None:
            self._background_depth = clean.copy()
            self._background_frames = 1
            return self._empty_result(clean, background_ready=False)

        if self._background_frames < self.motion.background_warmup_frames:
            self._update_warmup_background(clean)
            self._background_frames += 1
            if self._background_frames == self.motion.background_warmup_frames:
                # Missing depth usually means the empty scene is beyond our interaction range.
                self._background_depth[self._background_depth <= 0.0] = self.camera.depth_max
            return self._empty_result(
                clean,
                background_ready=self._background_frames >= self.motion.background_warmup_frames,
            )

        background = self._background_depth
        valid_current = clean > 0.0
        foreground_delta = np.where(valid_current, background - clean, 0.0)
        raw_mask = valid_current & (foreground_delta >= self.motion.motion_threshold)

        mask_u8 = raw_mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, self._kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, self._kernel)
        mask_u8 = self._remove_small_components(mask_u8)
        active_mask = mask_u8 > 0

        raw_sources = self._cluster_sources(
            clean,
            foreground_delta,
            active_mask,
            intrinsics,
            force_strength,
        )
        sources = self._stabilize_sources(raw_sources, dt)
        self._update_background(clean, active_mask)

        active_delta = foreground_delta[active_mask]
        return MotionResult(
            sources=sources,
            depth_small=clean,
            motion_mask=mask_u8,
            active_pixels=int(active_mask.sum()),
            average_delta=float(active_delta.mean()) if active_delta.size else 0.0,
            maximum_delta=float(active_delta.max()) if active_delta.size else 0.0,
            track_count=len(self._tracks),
            background_ready=True,
        )

    def _downsample_depth(self, depth_m: np.ndarray) -> np.ndarray:
        """Area-average valid metric depth instead of discarding most camera pixels."""
        valid = np.isfinite(depth_m) & (depth_m >= self.camera.depth_min)
        clipped = np.where(valid, np.minimum(depth_m, self.camera.depth_max), 0.0).astype(
            np.float32,
            copy=False,
        )
        target = (self.motion.downsample_width, self.motion.downsample_height)
        depth_average = cv2.resize(clipped, target, interpolation=cv2.INTER_AREA)
        valid_fraction = cv2.resize(valid.astype(np.float32), target, interpolation=cv2.INTER_AREA)
        return np.divide(
            depth_average,
            valid_fraction,
            out=np.zeros_like(depth_average),
            where=valid_fraction >= 0.35,
        ).astype(np.float32, copy=False)

    def _frame_dt(self, timestamp_ms: float | None) -> float:
        fallback = 1.0 / max(self.camera.fps, 1)
        if timestamp_ms is None or not np.isfinite(timestamp_ms):
            return fallback
        if self._previous_timestamp_ms is None:
            dt = fallback
        else:
            dt = (float(timestamp_ms) - self._previous_timestamp_ms) * 0.001
            dt = float(np.clip(dt, 1.0 / 240.0, 0.10))
        self._previous_timestamp_ms = float(timestamp_ms)
        return dt

    def _empty_result(self, depth: np.ndarray, background_ready: bool) -> MotionResult:
        return MotionResult(
            sources=ForceSources.empty(),
            depth_small=depth,
            motion_mask=np.zeros(depth.shape, dtype=np.uint8),
            active_pixels=0,
            average_delta=0.0,
            maximum_delta=0.0,
            track_count=0,
            background_ready=background_ready,
        )

    def _update_warmup_background(self, current: np.ndarray) -> None:
        assert self._background_depth is not None
        background = self._background_depth
        valid = current > 0.0
        missing = (background <= 0.0) & valid
        background[missing] = current[missing]
        common = (background > 0.0) & valid
        background[common] = np.maximum(background[common], current[common])

    def _update_background(self, current: np.ndarray, foreground: np.ndarray) -> None:
        assert self._background_depth is not None
        background = self._background_depth
        valid = current > 0.0
        missing = (background <= 0.0) & valid
        background[missing] = current[missing]

        common = (background > 0.0) & valid
        farther = common & (current > background)
        background[farther] += self.motion.background_recovery_alpha * (
            current[farther] - background[farther]
        )
        idle = common & ~farther & ~foreground
        background[idle] += self.motion.background_idle_alpha * (
            current[idle] - background[idle]
        )

    def _remove_small_components(self, mask_u8: np.ndarray) -> np.ndarray:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
        if count <= 1:
            return np.zeros_like(mask_u8)
        cleaned = np.zeros_like(mask_u8)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= self.motion.min_component_pixels:
                cleaned[labels == label] = 255
        return cleaned

    def _cluster_sources(
        self,
        depth: np.ndarray,
        foreground_delta: np.ndarray,
        active_mask: np.ndarray,
        intrinsics: DepthIntrinsics,
        force_strength: float,
    ) -> ForceSources:
        candidates: list[tuple[float, np.ndarray, float, float]] = []
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
                weights = np.maximum(
                    foreground_delta[global_y, global_x],
                    self.motion.motion_threshold,
                )
                weight_sum = float(weights.sum())
                center_x = float(np.dot(global_x, weights) / weight_sum)
                center_y = float(np.dot(global_y, weights) / weight_sum)
                source_depth = float(np.dot(depth[global_y, global_x], weights) / weight_sum)
                position = self._deproject_source(
                    center_x,
                    center_y,
                    source_depth,
                    width,
                    height,
                    intrinsics,
                )

                coverage = active_count / float((y1 - y0) * (x1 - x0))
                mean_separation = float(weights.mean())
                separation_scale = float(
                    np.clip(
                        mean_separation / max(self.motion.max_motion_delta, 1.0e-6),
                        0.0,
                        1.0,
                    )
                )
                strength = force_strength * (
                    0.34 + 0.42 * np.sqrt(coverage) + 0.18 * separation_scale
                )
                radius = self.simulation.force_radius * (
                    0.62 + 0.46 * np.sqrt(coverage)
                )
                score = strength * (0.35 + coverage)
                candidates.append((score, position, float(strength), float(radius)))

        if not candidates:
            return ForceSources.empty()

        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[: self.simulation.max_force_sources]
        count = len(candidates)
        return ForceSources(
            positions=np.stack([item[1] for item in candidates]).astype(np.float32),
            directions=np.zeros((count, 3), dtype=np.float32),
            strengths=np.asarray([item[2] for item in candidates], dtype=np.float32),
            radii=np.asarray([item[3] for item in candidates], dtype=np.float32),
        )

    def _deproject_source(
        self,
        x: float,
        y: float,
        depth: float,
        width: int,
        height: int,
        intrinsics: DepthIntrinsics,
    ) -> np.ndarray:
        full_u = (x + 0.5) * intrinsics.width / width - 0.5
        full_v = (y + 0.5) * intrinsics.height / height - 0.5
        camera_x = (full_u - intrinsics.ppx) / intrinsics.fx * depth
        camera_y = (full_v - intrinsics.ppy) / intrinsics.fy * depth
        if self.motion.mirror_x:
            camera_x = -camera_x
        return np.array(
            [
                camera_x * self.motion.scene_xy_scale,
                -camera_y * self.motion.scene_xy_scale,
                (self.motion.scene_depth_center - depth) * self.motion.scene_z_scale,
            ],
            dtype=np.float32,
        )

    def _stabilize_sources(self, sources: ForceSources, dt: float) -> ForceSources:
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_sources = set(range(len(sources)))
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            if len(sources):
                distances = np.linalg.norm(sources.positions - track.observed_position, axis=1)
                pairs.extend(
                    (float(distance), track_index, source_index)
                    for source_index, distance in enumerate(distances)
                    if float(distance) <= self.motion.source_match_distance
                )

        matches: list[tuple[int, int]] = []
        for _distance, track_index, source_index in sorted(pairs):
            if track_index in unmatched_tracks and source_index in unmatched_sources:
                unmatched_tracks.remove(track_index)
                unmatched_sources.remove(source_index)
                matches.append((track_index, source_index))

        for track_index, source_index in matches:
            track = self._tracks[track_index]
            measurement = sources.positions[source_index]
            measured_velocity = (measurement - track.observed_position) / max(dt, 1.0e-6)
            measured_speed = float(np.linalg.norm(measured_velocity))
            if measured_speed > self.motion.max_source_velocity:
                measured_velocity *= self.motion.max_source_velocity / measured_speed

            position_alpha = self.motion.source_position_alpha
            velocity_alpha = self.motion.source_velocity_alpha
            strength_alpha = self.motion.source_strength_alpha
            track.position += position_alpha * (measurement - track.position)
            track.velocity += velocity_alpha * (measured_velocity - track.velocity)
            track.observed_position = measurement.copy()

            speed = float(np.linalg.norm(track.velocity))
            if speed >= self.motion.velocity_deadzone:
                target_direction = track.velocity / speed
                blended = 0.58 * track.direction + 0.42 * target_direction
                track.direction = blended / max(float(np.linalg.norm(blended)), 1.0e-6)
            else:
                track.velocity *= 0.82

            speed_scale = min(speed / max(self.motion.velocity_reference, 1.0e-6), 1.5)
            approach_scale = min(
                max(float(track.velocity[2]), 0.0)
                / max(self.motion.velocity_reference, 1.0e-6),
                1.5,
            )
            target_strength = float(sources.strengths[source_index]) * (
                1.0
                + self.motion.velocity_boost * speed_scale
                + self.motion.approach_boost * approach_scale
            )
            track.strength += strength_alpha * (target_strength - track.strength)
            track.radius += strength_alpha * (
                float(sources.radii[source_index]) - track.radius
            )
            track.missed_frames = 0

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.missed_frames += 1
            track.position += track.velocity * (dt * 0.22)
            track.velocity *= 0.80
            track.strength *= self.motion.source_decay

        for source_index in unmatched_sources:
            position = sources.positions[source_index].copy()
            self._tracks.append(
                _ForceTrack(
                    position=position.copy(),
                    observed_position=position,
                    velocity=np.zeros(3, dtype=np.float32),
                    direction=np.zeros(3, dtype=np.float32),
                    strength=float(sources.strengths[source_index]) * 0.55,
                    radius=float(sources.radii[source_index]),
                )
            )

        self._tracks = [
            track
            for track in self._tracks
            if track.missed_frames <= self.motion.source_hold_frames and track.strength > 0.5
        ]
        self._tracks.sort(key=lambda track: track.strength, reverse=True)
        self._tracks = self._tracks[: self.simulation.max_force_sources]
        if not self._tracks:
            return ForceSources.empty()

        return ForceSources(
            positions=np.stack([track.position for track in self._tracks]).astype(np.float32),
            directions=np.stack([track.direction for track in self._tracks]).astype(np.float32),
            strengths=np.asarray([track.strength for track in self._tracks], dtype=np.float32),
            radii=np.asarray([track.radius for track in self._tracks], dtype=np.float32),
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
        mask_color[:, :, 1] = result.motion_mask
        combined = cv2.addWeighted(depth_color, 0.72, mask_color, 0.72, 0.0)
        combined = cv2.resize(combined, (640, 360), interpolation=cv2.INTER_NEAREST)
        if self.motion.mirror_x:
            combined = cv2.flip(combined, 1)
        status = (
            f"foreground={result.active_pixels} tracks={result.track_count}"
            if result.background_ready
            else f"learning empty scene {self._background_frames}/{self.motion.background_warmup_frames}"
        )
        cv2.putText(
            combined,
            status,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return combined
