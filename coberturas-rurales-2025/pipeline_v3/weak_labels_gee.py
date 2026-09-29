"""Etiquetas débiles 2025 contemporáneas desde Google Earth Engine (Fase 1).

Reemplaza la etiqueta CLC-2016 (desfasada 9 años) por un mapa de cobertura **2025**
alineado en el tiempo con la ortofoto. Fuente principal: **Dynamic World V1**
(10 m, 9 clases, deep learning, en GEE). Esto corrige de un golpe el sesgo temporal.

Flujo:
  1. ``export_dynamicworld``  — compone DW 2025 (moda anual de la banda 'label') sobre el
     AOI y lo descarga como GeoTIFF en EPSG:9377, 10 m (vía geedim, que tesela la descarga).
  2. ``homologate_to_model``  — remapea las 9 clases DW → clases del modelo v3
     (``legend.DYNAMICWORLD_TO_MODEL``). El resultado es el ráster de etiquetas débiles
     que alimenta el entrenamiento y estratifica el set de validación.

REQUISITO (una sola vez, lo hace el usuario): autenticarse en Earth Engine
    earthengine authenticate
y tener un proyecto de Cloud habilitado para EE (se pasa con --project).

Uso:
    # 1) exportar/descargar DW 2025 sobre el AOI
    python -m pipeline_v3.weak_labels_gee export \
        --ortho workspace_rural_aoi/data/prepared/ortho_2025_aoi.tif \
        --project MI_PROYECTO_EE \
        --out workspace_rural_aoi/data/labels/dw2025_label_10m.tif

    # 2) homologar a clases del modelo
    python -m pipeline_v3.weak_labels_gee homologate \
        --dw workspace_rural_aoi/data/labels/dw2025_label_10m.tif \
        --out workspace_rural_aoi/data/labels/weak_labels_dw2025_model.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from .legend import DYNAMICWORLD_TO_MODEL, IGNORE_INDEX, MODEL_LEGEND

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
# EE no puede parsear EPSG:9377 (MAGNA-SIRGAS 2018 / Origen-Nacional): se descarga en
# UTM 18N (EPSG:32618, zona de Bogotá) y se reproyecta localmente a la grilla de la orto
# en el paso de homologación.
EE_DOWNLOAD_CRS = "EPSG:32618"
TARGET_SCALE_M = 10.0


def _ortho_bounds_wgs84(ortho_path: Path) -> tuple[float, float, float, float]:
    """Límites del AOI en lon/lat (EPSG:4326) para construir la geometría EE."""
    from rasterio.warp import transform_bounds

    with rasterio.open(ortho_path) as src:
        left, bottom, right, top = src.bounds
        src_crs = src.crs
    return transform_bounds(src_crs, "EPSG:4326", left, bottom, right, top)


def export_dynamicworld(
    ortho_path: Path,
    project: str,
    out_path: Path,
    year: int = 2025,
    scale_m: float = TARGET_SCALE_M,
) -> Path:
    """Compone Dynamic World del año dado sobre el AOI y lo descarga como GeoTIFF."""
    import ee
    import geedim as gd

    ee.Initialize(project=project)

    w, s, e, n = _ortho_bounds_wgs84(ortho_path)
    aoi = ee.Geometry.Rectangle([w, s, e, n])

    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    col = (
        ee.ImageCollection(DW_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("label")
    )
    size = col.size().getInfo()
    if not size:
        raise RuntimeError(f"Sin imágenes Dynamic World para {year} en el AOI.")
    # Moda anual de la etiqueta (clase más frecuente por píxel a lo largo del año).
    # OJO: se desplaza +1 (clases 1-9) porque la clase 0 de DW es 'water' y geedim usa
    # 0 como nodata en la descarga — sin el offset, el agua se confunde con nodata.
    label = col.reduce(ee.Reducer.mode()).add(1).rename("label").toUint8().clip(aoi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = gd.MaskedImage(label)
    img.download(
        str(out_path),
        region=aoi,
        crs=EE_DOWNLOAD_CRS,
        scale=scale_m,
        dtype="uint8",
        overwrite=True,
    )
    print(f"Dynamic World {year}: {size} imágenes -> moda -> {out_path} ({EE_DOWNLOAD_CRS})")
    return out_path


def homologate_to_model(dw_path: Path, out_path: Path, ortho_path: Path | None = None) -> Path:
    """Remapea las clases Dynamic World (0-8) a las clases del modelo v3.

    Si se pasa ``ortho_path``, además reproyecta el resultado al CRS de la orto
    (EPSG:9377) en grilla de 10 m con vecino más cercano (correcto para categóricas),
    de modo que el ráster quede alineado con el resto del workspace.
    """
    # El ráster descargado trae las clases DW desplazadas +1 (1-9; 0 = nodata/enmascarado),
    # ver export_dynamicworld. La LUT descuenta el offset.
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for dw_id, model_id in DYNAMICWORLD_TO_MODEL.items():
        lut[dw_id + 1] = model_id

    with rasterio.open(dw_path) as src:
        profile = src.profile.copy()
        data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

    model = lut[data]
    model[data == 0] = IGNORE_INDEX

    if ortho_path is not None:
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform, reproject

        with rasterio.open(ortho_path) as o:
            dst_crs = o.crs
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, model.shape[1], model.shape[0],
            *rasterio.transform.array_bounds(model.shape[0], model.shape[1], src_transform),
            resolution=TARGET_SCALE_M,
        )
        warped = np.full((dst_h, dst_w), IGNORE_INDEX, dtype=np.uint8)
        reproject(
            source=model,
            destination=warped,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=IGNORE_INDEX,
            dst_nodata=IGNORE_INDEX,
            resampling=Resampling.nearest,
        )
        model = warped
        profile.update(crs=dst_crs, transform=dst_transform, width=dst_w, height=dst_h)

    profile.update(count=1, dtype="uint8", nodata=IGNORE_INDEX, compress="lzw")
    if profile["width"] >= 256 and profile["height"] >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    else:
        profile.update(tiled=False)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(model, 1)

    vals, counts = np.unique(model[model != IGNORE_INDEX], return_counts=True)
    dist = {MODEL_LEGEND[int(v)].name: int(c) for v, c in zip(vals, counts)}
    print(f"Etiquetas débiles homologadas -> {out_path}")
    print("Distribución por clase del modelo (px):")
    for name, c in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} {c:>12,}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Etiquetas débiles 2025 desde GEE (Dynamic World)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="Componer y descargar DW del AOI")
    pe.add_argument("--ortho", required=True, type=Path)
    pe.add_argument("--project", required=True, help="Proyecto de Cloud habilitado para EE")
    pe.add_argument("--year", type=int, default=2025)
    pe.add_argument("--out", required=True, type=Path)

    ph = sub.add_parser("homologate", help="Remapear DW -> clases del modelo v3")
    ph.add_argument("--dw", required=True, type=Path)
    ph.add_argument("--out", required=True, type=Path)
    ph.add_argument("--ortho", type=Path, default=None,
                    help="Si se pasa, reproyecta a la grilla/CRS de la orto (10 m, nearest)")

    args = ap.parse_args()
    if args.cmd == "export":
        export_dynamicworld(args.ortho, args.project, args.out, args.year)
    elif args.cmd == "homologate":
        homologate_to_model(args.dw, args.out, args.ortho)


if __name__ == "__main__":
    main()
