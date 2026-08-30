"""GPU-resident particle state and launch plumbing."""

import math

import numpy as np
import warp as wp

from config import SimulationConfig
from interaction.depth_motion import ForceSources
from simulation.kernels import integrate_particles, reset_particles


def _make_particle_field(count: int, size: tuple[float, float, float]) -> np.ndarray:
    """Create an exact-size, jittered structured slab with intentional rounded edges."""
    if count <= 0:
        raise ValueError("particle_count must be positive")

    base = (count / (1.875 * 1.25)) ** (1.0 / 3.0)
    nx = max(1, round(1.875 * base))
    ny = max(1, round(1.25 * base))
    nz = max(1, math.ceil(count / (nx * ny)))

    x = np.linspace(-0.5 * size[0], 0.5 * size[0], nx, dtype=np.float32)
    y = np.linspace(-0.5 * size[1], 0.5 * size[1], ny, dtype=np.float32)
    z = np.linspace(-0.5 * size[2], 0.5 * size[2], nz, dtype=np.float32)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)[:count]

    rng = np.random.default_rng(20260830)
    spacing = np.array(
        [
            size[0] / max(nx - 1, 1),
            size[1] / max(ny - 1, 1),
            size[2] / max(nz - 1, 1),
        ],
        dtype=np.float32,
    )
    grid += rng.uniform(-0.38, 0.38, size=grid.shape).astype(np.float32) * spacing

    # Gently round the rectangle without turning it into a sparse sphere.
    y_normalized = grid[:, 1] / max(size[1] * 0.5, 1.0e-6)
    z_normalized = grid[:, 2] / max(size[2] * 0.5, 1.0e-6)
    grid[:, 0] *= 1.0 - 0.10 * np.power(np.abs(y_normalized), 4)
    grid[:, 1] *= 1.0 - 0.06 * np.power(np.abs(z_normalized), 4)
    return np.ascontiguousarray(grid, dtype=np.float32)


def _make_particle_colors(positions: np.ndarray, size: tuple[float, float, float]) -> np.ndarray:
    y = np.clip(positions[:, 1] / size[1] + 0.5, 0.0, 1.0)
    z = np.clip(positions[:, 2] / size[2] + 0.5, 0.0, 1.0)
    colors = np.empty_like(positions)
    colors[:, 0] = 0.12 + 0.28 * z
    colors[:, 1] = 0.50 + 0.36 * y
    colors[:, 2] = 0.88 + 0.10 * (1.0 - y)
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


class ParticleSystem:
    def __init__(self, config: SimulationConfig, device: str = "cuda:0"):
        self.config = config
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(
                f"DepthForce requires a CUDA device for the MVP, but {self.device} was selected."
            )

        self.initial_positions = _make_particle_field(config.particle_count, config.field_size)
        self.colors = _make_particle_colors(self.initial_positions, config.field_size)
        self.positions = wp.array(self.initial_positions, dtype=wp.vec3, device=self.device)
        self.velocities = wp.zeros(config.particle_count, dtype=wp.vec3, device=self.device)
        self.rest_positions = wp.array(self.initial_positions, dtype=wp.vec3, device=self.device)

        max_sources = config.max_force_sources
        self._host_source_positions = np.zeros((max_sources, 3), dtype=np.float32)
        self._host_source_directions = np.zeros((max_sources, 3), dtype=np.float32)
        self._host_source_strengths = np.zeros(max_sources, dtype=np.float32)
        self._host_source_radii = np.zeros(max_sources, dtype=np.float32)
        self.source_positions = wp.zeros(max_sources, dtype=wp.vec3, device=self.device)
        self.source_directions = wp.zeros(max_sources, dtype=wp.vec3, device=self.device)
        self.source_strengths = wp.zeros(max_sources, dtype=float, device=self.device)
        self.source_radii = wp.zeros(max_sources, dtype=float, device=self.device)
        self.source_count = 0

    def set_force_sources(self, sources: ForceSources) -> None:
        count = min(len(sources), self.config.max_force_sources)
        self.source_count = count
        if count == 0:
            return

        self._host_source_positions[:count] = sources.positions[:count]
        self._host_source_directions[:count] = sources.directions[:count]
        self._host_source_strengths[:count] = sources.strengths[:count]
        self._host_source_radii[:count] = sources.radii[:count]
        self.source_positions.assign(self._host_source_positions)
        self.source_directions.assign(self._host_source_directions)
        self.source_strengths.assign(self._host_source_strengths)
        self.source_radii.assign(self._host_source_radii)

    def clear_force_sources(self) -> None:
        self.source_count = 0

    def step(self, dt: float, simulation_time: float) -> None:
        wp.launch(
            kernel=integrate_particles,
            dim=self.config.particle_count,
            inputs=[
                self.positions,
                self.velocities,
                self.rest_positions,
                self.source_positions,
                self.source_directions,
                self.source_strengths,
                self.source_radii,
                self.source_count,
                dt,
                self.config.damping,
                self.config.spring_strength,
                self.config.max_velocity,
                self.config.idle_drift,
                simulation_time,
            ],
            device=self.device,
        )

    def reset(self) -> None:
        wp.launch(
            kernel=reset_particles,
            dim=self.config.particle_count,
            inputs=[self.positions, self.velocities, self.rest_positions],
            device=self.device,
        )
        self.clear_force_sources()
