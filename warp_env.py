"""Prepare Warp's cache/compiler environment before importing :mod:`warp`."""

import os
from pathlib import Path
import shutil


def configure_warp_environment(project_root: Path) -> None:
    """Use local caching and avoid NVRTC's Windows non-ASCII temp-path bug."""
    temp_path = os.environ.get("TEMP", os.environ.get("TMP", ""))
    needs_ascii_workaround = os.name == "nt" and any(
        ord(character) > 127 for character in f"{temp_path}{project_root}"
    )
    if not needs_ascii_workaround:
        os.environ.setdefault("WARP_CACHE_PATH", str(project_root / ".warp-cache"))
        return

    if temp_path and not any(ord(character) > 127 for character in temp_path):
        ascii_temp = Path(temp_path) / "depthforce-warp"
    else:
        system_drive = os.environ.get("SystemDrive", "C:")
        ascii_temp = Path(system_drive + os.sep) / "temp" / "depthforce-warp"
    try:
        ascii_temp.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Warp will produce a precise NVRTC error later. The README also documents
        # how to set TEMP/TMP manually if this directory cannot be created.
        return
    os.environ["TEMP"] = str(ascii_temp)
    os.environ["TMP"] = str(ascii_temp)
    existing_cache = os.environ.get("WARP_CACHE_PATH", "")
    if not existing_cache or any(ord(character) > 127 for character in existing_cache):
        os.environ["WARP_CACHE_PATH"] = str(ascii_temp / "cache")


def configure_warp_native_path(warp_module) -> None:
    """Mirror Warp's small header tree when NVRTC cannot address its install path."""
    package_root = Path(warp_module.__file__).resolve().parent
    if os.name != "nt" or not any(ord(character) > 127 for character in str(package_root)):
        return

    temp_path = Path(os.environ.get("TEMP", ""))
    if not temp_path or any(ord(character) > 127 for character in str(temp_path)):
        return
    mirror_root = temp_path / f"warp-native-{warp_module.__version__}"
    source_native = package_root / "native"
    target_native = mirror_root / "native"
    try:
        if not (target_native / "builtin.h").is_file():
            shutil.copytree(source_native, target_native, dirs_exist_ok=True)
    except OSError:
        return

    # Warp 1.16 computes this module global at import time. Point only the
    # compiler include root at the ASCII mirror; the installed DLLs stay local.
    import warp._src.build as warp_build

    warp_build.warp_home = str(mirror_root)
