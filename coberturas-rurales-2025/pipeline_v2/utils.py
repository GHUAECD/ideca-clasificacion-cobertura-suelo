from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


IGNORE_INDEX = 255


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: dict, path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(payload: dict, path: Path) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def copy_if_needed(src: Path, dst: Path, overwrite: bool = False) -> dict:
    ensure_parent(dst)
    action = "skipped"
    if overwrite or not dst.exists() or dst.stat().st_size == 0:
        shutil.copy2(src, dst)
        action = "copied"
    return {
        "src": str(src),
        "dst": str(dst),
        "action": action,
        "bytes": int(dst.stat().st_size) if dst.exists() else 0,
    }


def align_single_band(aux_src, ref_src, window) -> np.ndarray:
    bounds = rasterio.windows.bounds(window, ref_src.transform)
    aux_window = aux_src.window(*bounds).round_offsets().round_lengths()
    aux_arr = aux_src.read(1, window=aux_window).astype(np.float32)
    dst = np.zeros((int(window.height), int(window.width)), dtype=np.float32)
    reproject(
        source=aux_arr,
        destination=dst,
        src_transform=aux_src.window_transform(aux_window),
        src_crs=aux_src.crs,
        dst_transform=ref_src.window_transform(window),
        dst_crs=ref_src.crs,
        resampling=Resampling.bilinear,
    )
    return dst


def feature_stack(
    ortho: np.ndarray,
    elevation: np.ndarray,
    pixel_size: float,
    elevation_min: float,
    elevation_max: float,
    slope_scale: float,
) -> np.ndarray:
    ortho = ortho.astype(np.float32)
    rgbnir = np.clip(ortho[:4] / 255.0, 0.0, 1.0)
    r, g, _b, nir = ortho[:4]
    ndvi = (nir - r) / np.clip(nir + r, 1.0, None)
    ndwi = (g - nir) / np.clip(g + nir, 1.0, None)
    ndvi = np.clip((ndvi + 1.0) * 0.5, 0.0, 1.0)
    ndwi = np.clip((ndwi + 1.0) * 0.5, 0.0, 1.0)

    elev = np.nan_to_num(elevation.astype(np.float32), nan=elevation_min)
    elev_norm = np.clip((elev - elevation_min) / max(elevation_max - elevation_min, 1.0), 0.0, 1.0)
    gy, gx = np.gradient(elev, pixel_size, pixel_size)
    slope = np.sqrt(gx * gx + gy * gy)
    slope_norm = np.clip(slope / max(slope_scale, 1e-6), 0.0, 1.0)
    return np.concatenate([rgbnir, ndvi[None], ndwi[None], elev_norm[None], slope_norm[None]], axis=0).astype(np.float32)


def cosine_weight(size: int, floor: float = 0.10) -> np.ndarray:
    one = np.hanning(size).astype(np.float32)
    two = np.outer(one, one)
    two = floor + (1.0 - floor) * two
    return two.astype(np.float32)
