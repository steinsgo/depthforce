# DepthForce

DepthForce is a native-Windows Python MVP in which motion from an Intel RealSense D435i becomes a small 3D force field that repels a dense NVIDIA Warp particle volume. The particle state remains on the CUDA device; only the downsampled camera force representation is processed on the CPU and uploaded each camera frame.

The synthetic mode is fully usable without a camera:

```powershell
python main.py --synthetic
```

## Hardware and platform

- Windows 10/11, native (not WSL for camera access)
- NVIDIA CUDA-capable GPU; the target/test GPU is an RTX 3060 Laptop GPU with 6 GB VRAM
- Intel RealSense D435i connected through a USB 3 port for live mode
- A working OpenGL display context

Tested locally on 2026-08-30 with:

- Windows 11 23H2 (build 22631)
- Python 3.10.6 in `.venv`
- NVIDIA driver 560.70 (driver reports CUDA 12.6)
- CUDA toolkit 12.4 installed; Warp 1.16.0 uses its bundled CUDA 12.9 runtime toolchain
- `warp-lang` 1.16.0, `pyrealsense2` 2.58.3.10794, NumPy 2.2.6, Pyglet 2.1.11, OpenCV 4.12.0

The D435i was not physically connected during this implementation session. SDK enumeration and the no-device path were validated, but live stream/interaction validation still requires the camera.

## Setup

Python 3.10 is used because it is a conservative shared target for the current Warp and RealSense Windows wheels.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No global Python environment is modified.

### Windows non-ASCII paths

Warp 1.16's NVRTC compiler can fail when its temp or header paths contain non-ASCII characters. `warp_env.py` detects this, moves Warp's compiler cache/temp files to `C:\temp\depthforce-warp`, and mirrors only the small Warp header tree there. It does not move the virtual environment or change the installed package.

If `C:\temp` is absent or not writable, choose any writable ASCII-only directory before launching:

```powershell
New-Item -ItemType Directory -Force C:\warp-temp
$env:TEMP = "C:\warp-temp"
$env:TMP = "C:\warp-temp"
$env:WARP_CACHE_PATH = "C:\warp-temp\cache"
```

## Staged validation

Run these in order:

```powershell
# A: compile and execute a tiny CUDA kernel
python scripts\test_warp.py

# B: static Warp OpenGL rendering, 150k GPU particles
python scripts\test_renderer.py

# C: force kernel, worst-case 96-source benchmark and displacement check
python scripts\test_simulation.py

# D: enumerate the D435i and stream 640x480 @ 30 FPS depth
python scripts\test_realsense.py

# E/F: synthetic depth-motion extraction and 3D clustering
python scripts\test_depth_motion.py
```

`test_realsense.py` exits with a clear `NO DEVICE` result when a camera is not present.

## Run

Without a camera:

```powershell
python main.py --synthetic
```

With a connected D435i:

```powershell
python main.py
```

Useful diagnostic variants:

```powershell
python main.py --synthetic --debug
python main.py --debug --camera-debug
python main.py --synthetic --particles 100000
```

The defaults are 150,000 particles, `cuda:0`, and a 1280x800 render window. `--frames N` is available for repeatable benchmarks and automated smoke tests.

## Controls

| Key | Action |
|---|---|
| `ESC` / `Q` | Quit |
| `R` | Reset positions and velocities |
| `SPACE` | Enable/disable camera or synthetic interaction |
| `D` | Toggle renderer/live console diagnostics |
| `+` / `-` | Increase/decrease force strength |

With `--camera-debug`, the OpenCV depth/motion window appears only while debug mode is enabled. The primary output remains the Warp particle window.

## How it works

```text
D435i 640x480 depth @ 30 FPS
        |
        v
160x120 metric depth + temporal smoothing + motion threshold
        |
        v
10x10 grid clustering -> at most 96 camera-space 3D sources
        |
        v
small CPU-to-GPU source upload
        |
        v
150k GPU particles: repulsion + inertia + damping + spring-to-rest
        |
        v
Warp OpenGLRenderer consumes the live Warp position array
```

The source mapping deliberately exaggerates small motion, adds extra strength for motion toward the camera, and uses smooth radial falloff. Particles retain momentum and then return gradually to their rest positions.

Repository layout:

```text
main.py                         application/render loop and controls
config.py                       visual, simulation, camera, and motion constants
camera/realsense.py             native RealSense depth wrapper
interaction/depth_motion.py     motion mask, deprojection, source clustering
simulation/kernels.py           Warp integration/reset kernels
simulation/particles.py         GPU state and small source uploads
rendering.py                    low-detail particle geometry compatibility
scripts/                        staged hardware and subsystem tests
```

## Measured performance

On the RTX 3060 Laptop GPU above:

- Static 150k renderer test: **108.7 average FPS** over 180 frames.
- Integrated 150k synthetic demo: **119–122 steady-state FPS**; **92.8 average FPS** over 600 frames when including the first 1.47-second kernel compilation.
- 150k particles against the maximum 96 force sources: **1,530-1,781 simulation steps/s** across two 300-step synchronized runs.

Warp 1.16 currently ignores `render_points(..., as_spheres=False)` and otherwise instantiates a 32x32 sphere (2,048 triangles) for each tiny point. `rendering.py` changes only this renderer instance to a 4x6 mesh (48 triangles), improving the measured static result from 8.4 to 108.7 FPS. The particle position buffer still flows directly from Warp CUDA to the renderer's registered OpenGL buffer. Initialization uses the already-existing CPU rest positions, so there is no full GPU particle readback in the real-time loop.

## Current limitations

- Live D435i behavior is implemented but not hardware-validated in this session because no device was connected.
- Camera-space to scene-space calibration is intentionally simple. The default 0.25-2.0 m interaction range and scene scale may need small adjustments for the installation distance.
- Source clustering is a lightweight fixed image grid rather than connected-components tracking, so one large moving body may generate many nearby sources.
- Particles use one visual color field and do not use RGB camera color.
- The effect is an artistic spring/repulsion system, not a physically accurate fluid simulation.

The single best next visual improvement is a short GPU trail/afterimage pass. It would make hand swipes and temporary cavities much easier to read without adding perception models or changing the camera pipeline.

Future extensions can include vortex/attract modes, shockwaves, depth echo, RGB-colored particles, body-silhouette spawning, SPH/fluid behavior, and audio reactivity.
