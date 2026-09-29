"""AOI efectivo = footprint de la ortofoto ∩ AOI oficial.

Adaptación de un script interno previo (``gdal_footprint_fix.py``, usado para
el boundary de otra ortoimagen) al área rural. Cambios respecto al original:

- Usa ``gdal.Footprint`` (GDAL >= 3.8) en lugar de máscara numpy a resolución completa
  + Polygonize: la orto rural pesa 60 GB (~25 Gpx x 4 bandas) y el método original
  requeriría ~25 GB de RAM. Se decima primero a ~5 m (suficiente para un límite) y se
  vectoriza con los mismos criterios del script original: presencia de datos en
  cualquier banda (union), área mínima 100.000 m² y suavizado por simplificación.
- Se omite la sección de arcpy (ImportMosaicDatasetGeometry/BuildBoundary): eso
  alimentaba el catálogo de mosaicos de ArcGIS, no aplica a este flujo.
- Añade la intersección con el AOI oficial (07_Límite del Proyecto), porque el AOI
  oficial es un poco más grande que la imagen: el área de trabajo efectiva es donde
  hay imagen Y está dentro del límite oficial.

Salidas:
  - GPKG con capas ``footprint`` (borde de datos de la orto) y ``aoi_efectivo``
    (footprint ∩ AOI oficial) en EPSG:9377.
  - Shapefile del ``aoi_efectivo`` (compatibilidad ArcGIS).

Uso:
    python -m pipeline_v3.aoi_footprint \
        --ortho "<RUTA_INSUMOS>/ortofoto_2025_0_5m.tif" \
        --aoi-oficial "<RUTA_INSUMOS>/AOI_BOGOTA_RURAL.shp" \
        --out workspace_rural_aoi/data/aoi/aoi_efectivo.gpkg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
from osgeo import gdal

gdal.UseExceptions()

# Criterios heredados del script original (gdal_footprint_fix.py)
AREA_MINIMA_M2 = 100_000      # descarta polígonos espurios < 10 ha
TOLERANCIA_SUAVIZADO_M = 2.0  # simplificación; 2 m acorde a la decimación de 5 m
DECIMATED_RES_M = 5.0         # resolución de trabajo para el footprint
TARGET_EPSG = 9377


def build_footprint(ortho_path: Path, decimated: Path):
    """Footprint de datos válidos (cualquier banda > 0) de la orto, decimado a ~5 m.

    Devuelve la geometría (shapely) del footprint en el CRS de la orto. La decimación
    se persiste en ``decimated`` y se reutiliza si ya existe (es el paso caro: una
    pasada completa sobre el TIF de 60 GB).
    """
    import numpy as np
    import rasterio
    from rasterio.features import shapes as raster_shapes
    from shapely.geometry import shape as to_shape
    from shapely.ops import unary_union

    if not decimated.exists():
        t0 = time.time()
        src = gdal.Open(str(ortho_path))
        gt = src.GetGeoTransform()
        factor = DECIMATED_RES_M / abs(gt[1])
        width = max(1, int(src.RasterXSize / factor))
        print(f"Decimando {src.RasterXSize}x{src.RasterYSize} px -> ~{DECIMATED_RES_M} m "
              f"({width} px de ancho)...")
        decimated.parent.mkdir(parents=True, exist_ok=True)
        gdal.Translate(
            str(decimated),
            src,
            options=gdal.TranslateOptions(
                width=width,
                height=0,  # preserva proporción
                resampleAlg="nearest",  # mantiene nítido el borde dato/no-dato
                creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
            ),
        )
        src = None
        print(f"  decimación: {time.time() - t0:.0f} s")
    else:
        print(f"Decimado existente, se reutiliza: {decimated}")

    t0 = time.time()
    print("Vectorizando footprint (mascara cualquier banda > 0)...")
    with rasterio.open(decimated) as ds:
        mask = np.zeros((ds.height, ds.width), dtype=bool)
        for b in range(1, ds.count + 1):
            mask |= ds.read(b) > 0  # criterio del script original
        transform = ds.transform

    polys = [
        to_shape(geom)
        for geom, val in raster_shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        if val == 1
    ]
    polys = [p for p in polys if p.area >= AREA_MINIMA_M2]
    if not polys:
        raise RuntimeError("No se encontraron polígonos de datos válidos.")
    footprint = unary_union(polys).simplify(TOLERANCIA_SUAVIZADO_M, preserve_topology=True)
    print(f"  footprint: {time.time() - t0:.0f} s | poligonos retenidos: {len(polys)}")
    return footprint


def effective_aoi(fp_geom, ortho_crs, aoi_oficial: Path, out_path: Path) -> None:
    fp_gdf = gpd.GeoDataFrame(geometry=[fp_geom], crs=ortho_crs).to_crs(TARGET_EPSG)
    oficial = gpd.read_file(aoi_oficial).to_crs(TARGET_EPSG)

    fp_geom = fp_gdf.union_all()
    of_geom = oficial.union_all()
    efectivo = fp_geom.intersection(of_geom)

    km2 = lambda g: g.area / 1e6  # noqa: E731
    print(f"AOI oficial   : {km2(of_geom):8.1f} km2")
    print(f"Footprint orto: {km2(fp_geom):8.1f} km2")
    print(f"AOI efectivo  : {km2(efectivo):8.1f} km2 (interseccion)")
    print(f"Oficial sin imagen: {km2(of_geom.difference(fp_geom)):.1f} km2")
    print(f"Imagen fuera del oficial: {km2(fp_geom.difference(of_geom)):.1f} km2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"name": ["footprint_orto"]}, geometry=[fp_geom], crs=TARGET_EPSG).to_file(
        out_path, layer="footprint", driver="GPKG")
    gpd.GeoDataFrame({"name": ["aoi_efectivo"]}, geometry=[efectivo], crs=TARGET_EPSG).to_file(
        out_path, layer="aoi_efectivo", driver="GPKG")
    shp = out_path.with_suffix(".shp")
    gpd.GeoDataFrame({"Name": ["aoi_efectivo"]}, geometry=[efectivo], crs=TARGET_EPSG).to_file(shp)
    print(f"Salidas: {out_path} (capas footprint, aoi_efectivo) y {shp}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AOI efectivo: footprint de la orto ∩ AOI oficial")
    ap.add_argument("--ortho", required=True, type=Path)
    ap.add_argument("--aoi-oficial", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--decimated", type=Path, default=None,
                    help="Ruta del TIF decimado persistente (default: junto a --out)")
    args = ap.parse_args()

    decimated = args.decimated or (args.out.parent / "ortho_fuente_decimada_5m.tif")
    footprint = build_footprint(args.ortho, decimated)

    import rasterio
    with rasterio.open(decimated) as ds:
        ortho_crs = ds.crs
    effective_aoi(footprint, ortho_crs, args.aoi_oficial, args.out)


if __name__ == "__main__":
    main()
