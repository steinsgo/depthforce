"""Central configuration for the small DepthForce MVP."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationConfig:
    particle_count: int = 150_000
    field_size: tuple[float, float, float] = (5.8, 3.5, 1.15)
    point_radius: float = 0.018
    spring_strength: float = 5.2
    damping: float = 2.15
    max_velocity: float = 5.5
    idle_drift: float = 0.028
    force_strength: float = 38.0
    force_radius: float = 0.78
    max_force_sources: int = 96
    max_step: float = 1.0 / 60.0


@dataclass(slots=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    depth_min: float = 0.25
    depth_max: float = 2.0


@dataclass(slots=True)
class MotionConfig:
    downsample_width: int = 160
    downsample_height: int = 120
    motion_threshold: float = 0.018
    max_motion_delta: float = 0.22
    depth_ema_alpha: float = 0.72
    cell_width: int = 10
    cell_height: int = 10
    min_active_pixels_per_cell: int = 5
    scene_xy_scale: float = 3.25
    scene_z_scale: float = 1.55
    scene_depth_center: float = 1.05
    approach_boost: float = 0.85


@dataclass(slots=True)
class RenderConfig:
    width: int = 1280
    height: int = 800
    target_fps: int = 60
    background_color: tuple[float, float, float] = (0.006, 0.009, 0.018)
    camera_pos: tuple[float, float, float] = (0.0, 0.25, 8.1)
    camera_front: tuple[float, float, float] = (0.0, -0.028, -1.0)
    camera_fov: float = 42.0


@dataclass(slots=True)
class AppConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
