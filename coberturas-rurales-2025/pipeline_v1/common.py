from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import mapping

gdal.UseExceptions()

MACRO_CLASSES = {
    0: "Artificial",
    1: "Cultivos",
    2: "Pastos_y_mosaicos",
    3: "Bosque",
    4: "Arbustal_y_secundaria",
    5: "Herbazal",
    6: "Frailejonal",
    7: "Humedal",
    8: "Suelo_desnudo_rocoso",
    9: "Agua",
}

CRITICAL_CLASSES = {"Bosque", "Herbazal", "Frailejonal", "Humedal"}

CONSERVATIVE_CLASSES = {
    "Bosque",
    "Arbustal_y_secundaria",
    "Herbazal",
    "Frailejonal",
    "Humedal",
}

FLEXIBLE_CLASSES = {"Cultivos", "Pastos_y_mosaicos"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def copy_sidecar_dataset(src: Path, dst: Path) -> list[Path]:
    ensure_parent(dst)
    copied: list[Path] = []
    for sidecar in src.parent.glob(f"{src.stem}.*"):
        target = dst.with_suffix(sidecar.suffix)
        shutil.copy2(sidecar, target)
        copied.append(target)
    return copied


def raster_metadata(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform),
            "bounds": list(src.bounds),
            "resolution": [src.res[0], src.res[1]],
        }


def vector_metadata(path: Path, layer: str | None = None) -> dict:
    gdf = gpd.read_file(path, layer=layer, rows=1)
    total = len(gpd.read_file(path, layer=layer, columns=[]))
    return {
        "path": str(path),
        "layer": layer,
        "features": total,
        "crs": gdf.crs.to_string() if gdf.crs else None,
        "bounds": list(gdf.total_bounds) if total else None,
    }


def build_manifest(rows: Iterable[dict], out_csv: Path) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    ensure_parent(out_csv)
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    return frame


def block_grid(bounds: tuple[float, float, float, float], block_size: float, crs: str) -> gpd.GeoDataFrame:
    from shapely.geometry import box

    xmin, ymin, xmax, ymax = bounds
    cols = int(math.ceil((xmax - xmin) / block_size))
    rows = int(math.ceil((ymax - ymin) / block_size))
    records = []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            x0 = xmin + col * block_size
            y0 = ymin + row * block_size
            x1 = min(x0 + block_size, xmax)
            y1 = min(y0 + block_size, ymax)
            records.append(
                {
                    "block_id": idx,
                    "row_id": row,
                    "col_id": col,
                    "geometry": box(x0, y0, x1, y1),
                }
            )
            idx += 1
    return gpd.GeoDataFrame(records, crs=crs)


def mode_ignore_negative(values: np.ndarray) -> int:
    valid = values[values >= 0]
    if valid.size == 0:
        return -1
    counts = np.bincount(valid.astype(np.int32))
    return int(np.argmax(counts))


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def reproject_to_match(
    source: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    resampling: Resampling,
) -> np.ndarray:
    dest = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=source,
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dest


def save_gpkg(gdf: gpd.GeoDataFrame, path: Path, layer: str = "data") -> None:
    ensure_parent(path)
    gdf.to_file(path, layer=layer, driver="GPKG")


def warp_raster(
    src: Path,
    dst: Path,
    dst_srs: str | None = None,
    cutline: Path | None = None,
    crop_to_cutline: bool = False,
    x_res: float | None = None,
    y_res: float | None = None,
    resample_alg: str = "bilinear",
    creation_options: list[str] | None = None,
    output_bounds: tuple[float, float, float, float] | None = None,
    width: int | None = None,
    height: int | None = None,
    target_aligned_pixels: bool = False,
) -> None:
    ensure_parent(dst)
    options = gdal.WarpOptions(
        dstSRS=dst_srs,
        cutlineDSName=str(cutline) if cutline else None,
        cropToCutline=crop_to_cutline,
        xRes=x_res,
        yRes=y_res,
        outputBounds=output_bounds,
        width=width,
        height=height,
        targetAlignedPixels=target_aligned_pixels,
        resampleAlg=resample_alg,
        creationOptions=creation_options or ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        multithread=True,
    )
    gdal.Warp(str(dst), str(src), options=options)


def translate_raster(src: Path, dst: Path, creation_options: list[str] | None = None) -> None:
    ensure_parent(dst)
    options = gdal.TranslateOptions(
        creationOptions=creation_options or ["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    )
    gdal.Translate(str(dst), str(src), options=options)


def normalized_change_vector(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    diff = img_b.astype(np.float32) - img_a.astype(np.float32)
    return np.linalg.norm(diff, axis=0)
