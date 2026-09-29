"""Fusión de etiquetas débiles: DW 2025 (base) + clases estables de CLC 2016 (Fase 1b).

Dynamic World 2025 es excelente para coberturas dinámicas y mayoritarias, pero
estructuralmente NO puede representar dos clases críticas del páramo de Sumapaz:

  - **Frailejonal**: no existe en la leyenda DW (cae en 'grass' → Pastos). Resultado:
    0 px en las etiquetas débiles, pese a ser la clase que el usuario más confirmó.
  - **Humedal**: DW lo da en cantidad ínfima (~2.900 px) frente a su presencia real.

Para estas dos clases —y solo estas— la interpretación CLC 2016 sí es defendible como
etiqueta 2025: el frailejonal crece ~1 cm/año y los humedales/turberas de páramo no
migran (a diferencia de cultivos o urbanización). Es la división de roles correcta:
DW para lo que cambia, CLC para lo estable que DW no distingue.

Este paso rasteriza los polígonos CLC de Frailejonal (321114) y Humedal (411/412/413x)
sobre la grilla de las etiquetas débiles, con una ligera erosión para no contaminar con
bordes, y los inyecta sobre la base DW (sobre-escribe lo que DW dijera ahí).

Uso:
    python -m pipeline_v3.label_fusion \
        --weak workspace_rural_aoi/data/labels/weak_labels_dw2025_model.tif \
        --clc-vector workspace_rural_aoi/data/prepared/coverage_2016_aoi.gpkg \
        --clc-mapping workspace_rural_aoi/config/class_mapping.csv \
        --out workspace_rural_aoi/data/labels/weak_labels_fused.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from .legend import IGNORE_INDEX, MACRO10_TO_MODEL, MODEL_LEGEND

# Clases estables a inyectar desde CLC: macro_id (v1) -> model_id (v3)
INJECT_MACROS = {6: MACRO10_TO_MODEL[6], 7: MACRO10_TO_MODEL[7]}  # Frailejonal, Humedal
EROSION_M = 10.0  # erosiona ~1 px a 10 m para evitar bordes ruidosos


def fuse(weak_path: Path, clc_vector: Path, clc_mapping: Path, out_path: Path,
         clc_layer: str = "coverage_2016") -> Path:
    with rasterio.open(weak_path) as src:
        profile = src.profile.copy()
        base = src.read(1)
        transform = src.transform
        crs = src.crs
        shape = base.shape
        res = abs(src.res[0])

    mapping = pd.read_csv(clc_mapping)[["CODIGO_ID", "macro_id"]].drop_duplicates()
    gdf = gpd.read_file(clc_vector, layer=clc_layer).merge(mapping, on="CODIGO_ID", how="left")
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    erosion_px = max(1, int(round(EROSION_M / res)))
    kernel = np.ones((erosion_px * 2 + 1, erosion_px * 2 + 1), dtype=np.uint8)

    fused = base.copy()
    injected = {}
    for macro_id, model_id in INJECT_MACROS.items():
        polys = gdf[gdf["macro_id"] == macro_id]
        if polys.empty:
            continue
        mask = rasterize(
            ((geom, 1) for geom in polys.geometry),
            out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
        )
        eroded = cv2.erode(mask, kernel, iterations=1)
        before = int((fused == model_id).sum())
        fused[eroded > 0] = model_id
        after = int((fused == model_id).sum())
        injected[MODEL_LEGEND[model_id].name] = {"px_antes": before, "px_despues": after}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(fused, 1)

    print(f"Etiquetas fusionadas -> {out_path}")
    print("Inyección desde CLC 2016 (clases estables):")
    for name, st in injected.items():
        print(f"  {name:20s} {st['px_antes']:>10,} -> {st['px_despues']:>10,} px")
    print("\nDistribución final por clase del modelo:")
    vals, counts = np.unique(fused[fused != IGNORE_INDEX], return_counts=True)
    for v, c in sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: -kv[1]):
        print(f"  {MODEL_LEGEND[int(v)].name:24s} {c:>12,}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fusión DW 2025 + clases estables CLC 2016")
    ap.add_argument("--weak", required=True, type=Path)
    ap.add_argument("--clc-vector", required=True, type=Path)
    ap.add_argument("--clc-mapping", required=True, type=Path)
    ap.add_argument("--clc-layer", default="coverage_2016")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    fuse(args.weak, args.clc_vector, args.clc_mapping, args.out, args.clc_layer)


if __name__ == "__main__":
    main()
