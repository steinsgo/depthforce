"""Validate persistent foreground extraction, metric tracking, and interaction presets."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from camera import DepthIntrinsics
from config import AppConfig, apply_interaction_preset
from interaction import DepthMotionExtractor


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    config = AppConfig()
    config.motion.background_warmup_frames = 4
    extractor = DepthMotionExtractor(config.camera, config.motion, config.simulation)
    intrinsics = DepthIntrinsics(
        width=640,
        height=360,
        fx=385.0,
        fy=385.0,
        ppx=320.0,
        ppy=180.0,
    )
    background = np.full((360, 640), 1.50, dtype=np.float32)
    timestamp = 0.0

    for _ in range(config.motion.background_warmup_frames):
        result = extractor.process(
            background,
            intrinsics,
            config.simulation.force_strength,
            timestamp,
        )
        timestamp += 1000.0 / config.camera.fps
    require(result.background_ready, "background model did not finish warmup")
    require(len(result.sources) == 0, "empty warmup scene produced force sources")

    hand = background.copy()
    hand[95:275, 180:300] = 0.80
    appeared = extractor.process(
        hand,
        intrinsics,
        config.simulation.force_strength,
        timestamp,
    )
    timestamp += 1000.0 / config.camera.fps
    require(appeared.active_pixels > 0, "foreground silhouette was not detected")
    require(len(appeared.sources) > 0, "foreground silhouette produced no force sources")

    stationary = extractor.process(
        hand,
        intrinsics,
        config.simulation.force_strength,
        timestamp,
    )
    timestamp += 1000.0 / config.camera.fps
    require(
        stationary.active_pixels > 0 and len(stationary.sources) > 0,
        "stationary foreground did not persist",
    )

    shifted = background.copy()
    shifted[95:275, 196:316] = 0.80
    lateral = extractor.process(
        shifted,
        intrinsics,
        config.simulation.force_strength,
        timestamp,
    )
    timestamp += 1000.0 / config.camera.fps
    moving_directions = lateral.sources.directions[
        np.linalg.norm(lateral.sources.directions, axis=1) > 0.1
    ]
    require(len(moving_directions) > 0, "tracked foreground did not generate XYZ velocity")
    require(
        float(np.median(moving_directions[:, 0])) < 0.0,
        "default mirror did not map right-image motion to negative scene X",
    )

    closer = shifted.copy()
    closer[95:275, 196:316] = 0.70
    approach = extractor.process(
        closer,
        intrinsics,
        config.simulation.force_strength,
        timestamp,
    )
    timestamp += 1000.0 / config.camera.fps
    require(
        bool(np.any(approach.sources.directions[:, 2] > 0.05)),
        "approaching foreground did not generate positive scene-Z velocity",
    )

    fading = extractor.process(
        background,
        intrinsics,
        config.simulation.force_strength,
        timestamp,
    )
    require(len(fading.sources) > 0, "force tracks popped off instead of fading")
    require(
        float(fading.sources.strengths.max()) < float(approach.sources.strengths.max()),
        "missing force tracks did not decay",
    )

    close_config = AppConfig()
    close_config.motion.background_warmup_frames = 1
    close_extractor = DepthMotionExtractor(
        close_config.camera,
        close_config.motion,
        close_config.simulation,
    )
    close_extractor.process(background, intrinsics, close_config.simulation.force_strength, 0.0)
    valid_near = background.copy()
    valid_near[120:240, 240:400] = 0.17
    near_result = close_extractor.process(
        valid_near,
        intrinsics,
        close_config.simulation.force_strength,
        1000.0 / close_config.camera.fps,
    )
    require(len(near_result.sources) > 0, "0.17 m foreground was rejected")

    balanced = AppConfig()
    apply_interaction_preset(balanced, "balanced")
    punchy = AppConfig()
    apply_interaction_preset(punchy, "punchy")
    require(
        punchy.simulation.force_strength > balanced.simulation.force_strength,
        "Punchy preset is not stronger than Balanced",
    )
    require(
        punchy.simulation.direction_mix > balanced.simulation.direction_mix,
        "Punchy preset does not emphasize XYZ travel direction",
    )

    print(
        "PASS: persistent foreground stays active while stationary, tracks lateral and "
        "approach XYZ velocity, fades smoothly, accepts 0.17 m depth, and exposes "
        "Balanced/Punchy presets."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
