"""Thin, synchronous Intel RealSense depth-camera wrapper."""

from dataclasses import dataclass

import numpy as np


class RealSenseUnavailableError(RuntimeError):
    """Raised when the SDK or a usable depth device is unavailable."""


@dataclass(frozen=True, slots=True)
class RealSenseDeviceInfo:
    name: str
    serial: str
    firmware: str


@dataclass(frozen=True, slots=True)
class DepthIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(slots=True)
class DepthFrame:
    depth_m: np.ndarray
    timestamp_ms: float


def _load_sdk():
    try:
        import pyrealsense2 as rs
    except (ImportError, RuntimeError) as exc:
        raise RealSenseUnavailableError(
            "pyrealsense2 could not be loaded. Install requirements in the local virtual environment."
        ) from exc
    return rs


def list_realsense_devices() -> list[RealSenseDeviceInfo]:
    """Return RealSense devices currently visible to the native Windows SDK."""
    rs = _load_sdk()
    try:
        context = rs.context()
        devices = context.query_devices()
        return [
            RealSenseDeviceInfo(
                name=device.get_info(rs.camera_info.name),
                serial=device.get_info(rs.camera_info.serial_number),
                firmware=device.get_info(rs.camera_info.firmware_version),
            )
            for device in devices
        ]
    except RuntimeError as exc:
        raise RealSenseUnavailableError(f"RealSense device enumeration failed: {exc}") from exc


class RealSenseCamera:
    """Own a RealSense pipeline and expose metric depth frames."""

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        fps: int = 60,
        visual_preset: str = "hand",
        spatial_filter_enabled: bool = True,
        spatial_filter_magnitude: float = 1.0,
        spatial_filter_alpha: float = 0.65,
        spatial_filter_delta: float = 20.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.visual_preset = visual_preset
        self.spatial_filter_enabled = spatial_filter_enabled
        self.spatial_filter_magnitude = spatial_filter_magnitude
        self.spatial_filter_alpha = spatial_filter_alpha
        self.spatial_filter_delta = spatial_filter_delta
        self.depth_scale = 0.0
        self.active_visual_preset = ""
        self.intrinsics: DepthIntrinsics | None = None
        self.device_info: RealSenseDeviceInfo | None = None
        self._rs = None
        self._context = None
        self._pipeline = None
        self._spatial_filter = None
        self._started = False

    def start(self) -> None:
        rs = _load_sdk()
        try:
            self._context = rs.context()
            devices = self._context.query_devices()
            if len(devices) == 0:
                raise RealSenseUnavailableError(
                    "No Intel RealSense device was found by the native Windows SDK."
                )

            device = devices[0]
            self.device_info = RealSenseDeviceInfo(
                name=device.get_info(rs.camera_info.name),
                serial=device.get_info(rs.camera_info.serial_number),
                firmware=device.get_info(rs.camera_info.firmware_version),
            )

            pipeline = rs.pipeline(self._context)
            pipeline_config = rs.config()
            pipeline_config.enable_device(self.device_info.serial)
            pipeline_config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )
            profile = pipeline.start(pipeline_config)

            depth_sensor = profile.get_device().first_depth_sensor()
            self._apply_visual_preset(depth_sensor, rs)
            self.depth_scale = float(depth_sensor.get_depth_scale())
            if depth_sensor.supports(rs.option.visual_preset):
                preset_value = depth_sensor.get_option(rs.option.visual_preset)
                self.active_visual_preset = depth_sensor.get_option_value_description(
                    rs.option.visual_preset,
                    preset_value,
                )
            video_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = video_profile.get_intrinsics()
            self.intrinsics = DepthIntrinsics(
                width=intr.width,
                height=intr.height,
                fx=float(intr.fx),
                fy=float(intr.fy),
                ppx=float(intr.ppx),
                ppy=float(intr.ppy),
            )
            self._rs = rs
            self._pipeline = pipeline
            if self.spatial_filter_enabled:
                spatial_filter = rs.spatial_filter()
                spatial_filter.set_option(
                    rs.option.filter_magnitude, self.spatial_filter_magnitude
                )
                spatial_filter.set_option(
                    rs.option.filter_smooth_alpha, self.spatial_filter_alpha
                )
                spatial_filter.set_option(
                    rs.option.filter_smooth_delta, self.spatial_filter_delta
                )
                spatial_filter.set_option(rs.option.holes_fill, 0.0)
                self._spatial_filter = spatial_filter
            self._started = True
        except RealSenseUnavailableError:
            raise
        except RuntimeError as exc:
            self.stop()
            raise RealSenseUnavailableError(
                f"Could not start {self.width}x{self.height}@{self.fps} depth streaming: {exc}"
            ) from exc

    def poll(self) -> DepthFrame | None:
        """Return a new frame when available, without blocking the render loop."""
        if not self._started or self._pipeline is None:
            raise RuntimeError("RealSenseCamera.start() must be called before poll().")

        try:
            frames = self._pipeline.poll_for_frames()
            if not frames:
                return None
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            depth_u16 = np.asanyarray(self._filter_depth(depth_frame).get_data())
            depth_m = depth_u16.astype(np.float32) * self.depth_scale
            return DepthFrame(depth_m=depth_m, timestamp_ms=float(depth_frame.get_timestamp()))
        except RuntimeError:
            # A transient USB/frame error should not take down the visual loop.
            return None

    def wait(self, timeout_ms: int = 2500) -> DepthFrame | None:
        """Wait for one frame; intended for the standalone camera test."""
        if not self._started or self._pipeline is None:
            raise RuntimeError("RealSenseCamera.start() must be called before wait().")
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            depth_u16 = np.asanyarray(self._filter_depth(depth_frame).get_data())
            return DepthFrame(
                depth_m=depth_u16.astype(np.float32) * self.depth_scale,
                timestamp_ms=float(depth_frame.get_timestamp()),
            )
        except RuntimeError:
            return None

    def stop(self) -> None:
        if self._started and self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
        self._started = False
        self._pipeline = None
        self._spatial_filter = None

    def _filter_depth(self, depth_frame):
        if self._spatial_filter is None:
            return depth_frame
        return self._spatial_filter.process(depth_frame).as_depth_frame()

    def _apply_visual_preset(self, depth_sensor, rs) -> None:
        if not self.visual_preset:
            return
        if not depth_sensor.supports(rs.option.visual_preset):
            raise RealSenseUnavailableError(
                f"This depth sensor does not support the requested '{self.visual_preset}' preset."
            )
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        requested = self.visual_preset.casefold()
        for value in range(int(preset_range.min), int(preset_range.max) + 1):
            description = depth_sensor.get_option_value_description(
                rs.option.visual_preset,
                float(value),
            )
            if description.casefold() == requested:
                depth_sensor.set_option(rs.option.visual_preset, float(value))
                return
        raise RealSenseUnavailableError(
            f"RealSense visual preset '{self.visual_preset}' is not available."
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
