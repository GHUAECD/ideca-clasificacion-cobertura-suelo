"""Re-recorte de la ortofoto y el MDT al AOI efectivo (paso 2 del flujo final).

En la ejecución de junio de 2026 este paso se hizo de forma interactiva con
``pipeline_v1.common.warp_raster``; este script lo deja reproducible con los mismos
parámetros documentados:

- Cutline = capa ``aoi_efectivo`` (salida de ``pipeline_v3.aoi_footprint``) con un
  buffer de 128 m, para que la inferencia por teselas tenga contexto en el borde.
- Píxeles alineados a la grilla (``targetAlignedPixels``): 0,5 m para la orto
  (vecino más cercano) y 10 m para el MDT (bilineal). EPSG:9377.

Verificación (sept-2026): con el ``aoi_efectivo`` de la entrega, este cálculo da
exactamente la extensión y el tamaño de los insumos usados
(orto 83.984 x 179.491 px; MDT 4.200 x 8.975 px).

Uso:
    python scripts/recortar_insumos.py \
        --ortho "<RUTA_INSUMOS>/ortofoto_2025_0_5m.tif" \
        --mdt "<RUTA_INSUMOS>/mdt_10m.tif" \
        --aoi workspace_rural_aoi/data/aoi/aoi_efectivo.gpkg \
        --out-dir workspace_rural_aoi/data/prepared
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v1.common import warp_raster  # noqa: E402

TARGET_SRS = "EPSG:9377"
BUFFER_M = 128.0


def buffered_cutline(aoi_path: Path, aoi_layer: str, tmp_dir: Path) -> Path:
    aoi = gpd.read_file(aoi_path, layer=aoi_layer).to_crs(9377)
    cut = gpd.GeoDataFrame(geometry=[aoi.union_all().buffer(BUFFER_M)], crs=9377)
    path = tmp_dir / "cutline_aoi_buffer.gpkg"
    cut.to_file(path, driver="GPKG")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-recorte de orto y MDT al AOI efectivo + 128 m")
    ap.add_argument("--ortho", type=Path, help="Ortofoto fuente RGB+NIR (0,5 m)")
    ap.add_argument("--mdt", type=Path, help="MDT fuente (10 m)")
    ap.add_argument("--aoi", required=True, type=Path)
    ap.add_argument("--aoi-layer", default="aoi_efectivo")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    if not (args.ortho or args.mdt):
        raise SystemExit("Indique --ortho, --mdt o ambos.")

    with tempfile.TemporaryDirectory() as tmp:
        cutline = buffered_cutline(args.aoi, args.aoi_layer, Path(tmp))
        jobs = []
        if args.ortho:
            jobs.append((args.ortho, args.out_dir / "ortho_2025_aoi_efectivo.tif", 0.5, "near"))
        if args.mdt:
            jobs.append((args.mdt, args.out_dir / "mdt_2025_aoi_efectivo.tif", 10.0, "bilinear"))
        for src, dst, res, resample in jobs:
            print(f"{src.name} -> {dst} ({res} m, {resample})", flush=True)
            # width/height = 0: GDAL 3.10 serializa None como literal 'None' y falla.
            warp_raster(src, dst, dst_srs=TARGET_SRS, cutline=cutline, crop_to_cutline=True,
                        x_res=res, y_res=res, resample_alg=resample, width=0, height=0,
                        target_aligned_pixels=True)


if __name__ == "__main__":
    main()
