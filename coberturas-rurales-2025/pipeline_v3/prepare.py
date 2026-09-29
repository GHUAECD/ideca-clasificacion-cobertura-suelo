"""Preparación de entrenamiento v3: bloques con splits espaciales + índice de chips.

Sobre los insumos definitivos (orto/MDT re-recortados al ``aoi_efectivo``):

1. Malla de bloques (1.024 m) recortada al AOI efectivo, con splits train/val/test
   asignados por bloque (no por chip) para evitar fuga espacial entre splits.
2. Índice de chips de 512 px (256 m): un chip es válido si su área dentro del AOI
   supera un umbral (la validez de datos la garantiza el footprint: dentro del AOI
   efectivo siempre hay imagen) y si su ventana de etiquetas débiles DW 2025 tiene
   cobertura suficiente. Por chip se guardan los conteos de clase del prior (para el
   sampler balanceado del entrenamiento).

Las etiquetas débiles (10 m) NO se pre-remuestrean a 0,5 m: el Dataset de
entrenamiento las alinea por ventana al vuelo (nearest), igual que v2 hacía con el
MDT (``pipeline_v2.utils.align_single_band``). Eso evita un intermedio de 15 Gpx.

Uso:
    python -m pipeline_v3.prepare \
        --ortho workspace_rural_aoi/data/prepared/ortho_2025_aoi_efectivo.tif \
        --labels workspace_rural_aoi/data/labels/weak_labels_dw2025_model.tif \
        --aoi workspace_rural_aoi/data/aoi/aoi_efectivo.gpkg \
        --out-dir workspace_rural_aoi/data/intermediate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .legend import IGNORE_INDEX, num_model_classes

BLOCK_SIZE_M = 1024.0
CHIP_PX = 512
CHIP_STRIDE_PX = 512
MIN_AOI_FRACTION = 0.5      # área mínima del chip dentro del AOI
MIN_LABEL_FRACTION = 0.3    # fracción mínima de píxeles DW etiquetados en el chip
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def build_blocks(aoi_path: Path, aoi_layer: str, crs: str, seed: int) -> gpd.GeoDataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline_v1.common import block_grid

    aoi = gpd.read_file(aoi_path, layer=aoi_layer).to_crs(crs)
    grid = block_grid(tuple(aoi.total_bounds), BLOCK_SIZE_M, crs)
    grid = gpd.overlay(grid, aoi[["geometry"]], how="intersection")
    grid["block_area_m2"] = grid.geometry.area
    grid = grid[grid["block_area_m2"] > 0].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(grid))
    train_cut = int(len(grid) * SPLITS["train"])
    val_cut = train_cut + int(len(grid) * SPLITS["val"])
    split = np.empty(len(grid), dtype=object)
    split[perm[:train_cut]] = "train"
    split[perm[train_cut:val_cut]] = "val"
    split[perm[val_cut:]] = "test"
    grid["split"] = split
    return grid


def read_label_window_aligned(label_src, ortho_src, win: Window) -> np.ndarray:
    """Etiquetas DW (10 m) alineadas a la ventana de la orto (0,5 m), nearest."""
    bounds = rasterio.windows.bounds(win, ortho_src.transform)
    dst = np.full((int(win.height), int(win.width)), IGNORE_INDEX, dtype=np.uint8)
    src_win = from_bounds(*bounds, transform=label_src.transform)
    src_win = src_win.round_offsets().round_lengths()
    if src_win.width <= 0 or src_win.height <= 0:
        return dst
    data = label_src.read(1, window=src_win, boundless=True, fill_value=IGNORE_INDEX)
    reproject(
        source=data,
        destination=dst,
        src_transform=label_src.window_transform(src_win),
        src_crs=label_src.crs,
        dst_transform=ortho_src.window_transform(win),
        dst_crs=ortho_src.crs,
        src_nodata=IGNORE_INDEX,
        dst_nodata=IGNORE_INDEX,
        resampling=Resampling.nearest,
    )
    return dst


def build_chip_index(
    ortho_path: Path,
    labels_path: Path,
    blocks: gpd.GeoDataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    n_classes = num_model_classes()
    records: list[dict] = []
    t0 = time.time()

    with rasterio.open(ortho_path) as ortho, rasterio.open(labels_path) as labels:
        # etiquetas en memoria a 10 m (pequeñas) para conteos rápidos por chip
        for bi, block in blocks.iterrows():
            minx, miny, maxx, maxy = block.geometry.bounds
            row0, col0 = ortho.index(minx, maxy)
            row1, col1 = ortho.index(maxx, miny)
            row_start, row_stop = max(0, min(row0, row1)), min(ortho.height, max(row0, row1))
            col_start, col_stop = max(0, min(col0, col1)), min(ortho.width, max(col0, col1))

            for row_off in range(row_start, max(row_start, row_stop - CHIP_PX + 1), CHIP_STRIDE_PX):
                for col_off in range(col_start, max(col_start, col_stop - CHIP_PX + 1), CHIP_STRIDE_PX):
                    win = Window(col_off, row_off, CHIP_PX, CHIP_PX)
                    if win.row_off + win.height > ortho.height or win.col_off + win.width > ortho.width:
                        continue
                    wb = rasterio.windows.bounds(win, ortho.transform)
                    from shapely.geometry import box
                    chip_geom = box(*wb)
                    frac_aoi = chip_geom.intersection(block.geometry).area / chip_geom.area
                    if frac_aoi < MIN_AOI_FRACTION:
                        continue

                    # conteos de clase del prior en la ventana (a 10 m, barato)
                    lab_win = from_bounds(*wb, transform=labels.transform).round_offsets().round_lengths()
                    lab = labels.read(1, window=lab_win, boundless=True, fill_value=IGNORE_INDEX)
                    valid = lab != IGNORE_INDEX
                    frac_label = float(valid.mean()) if lab.size else 0.0
                    if frac_label < MIN_LABEL_FRACTION:
                        continue
                    counts = np.bincount(lab[valid].astype(np.int64), minlength=n_classes)

                    rec = {
                        "block_id": int(block["block_id"]),
                        "split": block["split"],
                        "row_off": int(row_off),
                        "col_off": int(col_off),
                        "size_px": CHIP_PX,
                        "frac_aoi": round(frac_aoi, 4),
                        "frac_label": round(frac_label, 4),
                        "dominant_class": int(np.argmax(counts)) if counts.sum() else -1,
                    }
                    for c in range(n_classes):
                        rec[f"class_{c}_px10m"] = int(counts[c])
                    records.append(rec)

            if (bi + 1) % 200 == 0:
                print(f"  bloques {bi + 1}/{len(blocks)} | chips {len(records)} | {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Preparación v3: bloques + chips")
    ap.add_argument("--ortho", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--aoi", required=True, type=Path)
    ap.add_argument("--aoi-layer", default="aoi_efectivo")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with rasterio.open(args.ortho) as o:
        crs = o.crs

    blocks = build_blocks(args.aoi, args.aoi_layer, crs, args.seed)
    blocks_path = args.out_dir / "v3_blocks.gpkg"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks.to_file(blocks_path, layer="blocks", driver="GPKG")
    print(f"bloques: {len(blocks)} -> {blocks_path}")
    print(blocks["split"].value_counts().to_string())

    chips = build_chip_index(args.ortho, args.labels, blocks, args.out_dir / "v3_chips.csv")
    print(f"chips: {len(chips)} -> {args.out_dir / 'v3_chips.csv'}")
    print(chips.groupby("split").size().to_string())
    print("clase dominante (chips):")
    print(chips["dominant_class"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
