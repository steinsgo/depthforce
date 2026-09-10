"""Central configuration for the small DepthForce MVP."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationConfig:
    particle_count: int = 150_000
    field_size: tuple[float, float, float] = (5.8, 3.5, 1.15)
    point_radius: float = 0.018
    spring_strength: float = 4.6
    damping: float = 2.65
    max_velocity: float = 5.5
    idle_drift: float = 0.028
    force_strength: float = 52.0
    force_radius: float = 0.82
    direction_mix: float = 0.32
    max_force_sources: int = 96
    max_step: float = 1.0 / 60.0


@dataclass(slots=True)
class CameraConfig:
    width: int = 640
    height: int = 360
    fps: int = 60
    visual_preset: str = "hand"
    depth_min: float = 0.16
    depth_max: float = 1.8
    spatial_filter_enabled: bool = True
    spatial_filter_magnitude: float = 1.0
    spatial_filter_alpha: float = 0.65
    spatial_filter_delta: float = 20.0


@dataclass(slots=True)
class MotionConfig:
    downsample_width: int = 160
    downsample_height: int = 90
    motion_threshold: float = 0.045
    max_motion_delta: float = 0.35
    background_warmup_frames: int = 20
    background_recovery_alpha: float = 0.42
    background_idle_alpha: float = 0.012
    cell_width: int = 10
    cell_height: int = 10
    min_active_pixels_per_cell: int = 4
    min_component_pixels: int = 24
    scene_xy_scale: float = 3.25
    scene_z_scale: float = 1.55
    scene_depth_center: float = 1.05
    approach_boost: float = 0.85
    mirror_x: bool = True
    source_match_distance: float = 0.55
    source_position_alpha: float = 0.30
    source_velocity_alpha: float = 0.26
    source_strength_alpha: float = 0.28
    source_decay: float = 0.88
    source_hold_frames: int = 12
    velocity_deadzone: float = 0.04
    velocity_reference: float = 1.4
    velocity_boost: float = 0.55
    max_source_velocity: float = 6.0


@dataclass(slots=True)
class RenderConfig:
    width: int = 1280
    height: int = 800
    maximized: bool = True
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


@dataclass(frozen=True, slots=True)
class InteractionPreset:
    force_strength: float
    force_radius: float
    spring_strength: float
    damping: float
    direction_mix: float
    foreground_threshold: float
    position_alpha: float
    velocity_alpha: float
    velocity_boost: float
    source_decay: float
    source_hold_frames: int


INTERACTION_PRESETS: dict[str, InteractionPreset] = {
    "balanced": InteractionPreset(
        force_strength=52.0,
        force_radius=0.82,
        spring_strength=4.6,
        damping=2.65,
        direction_mix=0.32,
        foreground_threshold=0.045,
        position_alpha=0.30,
        velocity_alpha=0.26,
        velocity_boost=0.55,
        source_decay=0.88,
        source_hold_frames=12,
    ),
    "punchy": InteractionPreset(
        force_strength=76.0,
        force_radius=0.94,
        spring_strength=5.4,
        damping=2.05,
        direction_mix=0.48,
        foreground_threshold=0.035,
        position_alpha=0.48,
        velocity_alpha=0.44,
        velocity_boost=1.05,
        source_decay=0.82,
        source_hold_frames=8,
    ),
}


def apply_interaction_preset(config: AppConfig, name: str) -> None:
    """Apply one coherent camera-interaction feel without changing hardware settings."""
    try:
        preset = INTERACTION_PRESETS[name.casefold()]
    except KeyError as exc:
        choices = ", ".join(INTERACTION_PRESETS)
        raise ValueError(f"Unknown interaction preset '{name}'. Choose: {choices}.") from exc

    simulation = config.simulation
    motion = config.motion
    simulation.force_strength = preset.force_strength
    simulation.force_radius = preset.force_radius
    simulation.spring_strength = preset.spring_strength
    simulation.damping = preset.damping
    simulation.direction_mix = preset.direction_mix
    motion.motion_threshold = preset.foreground_threshold
    motion.source_position_alpha = preset.position_alpha
    motion.source_velocity_alpha = preset.velocity_alpha
    motion.velocity_boost = preset.velocity_boost
    motion.source_decay = preset.source_decay
    motion.source_hold_frames = preset.source_hold_frames
