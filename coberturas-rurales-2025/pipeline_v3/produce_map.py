"""Producto final: mapa de cobertura 2025 = prior fusionado regularizado con SAM 2 (Fase 4).

Para cada tesela de la orto:
  1. SAM 2 segmenta la imagen 0,5 m (objetos de cobertura, fronteras reales).
  2. Cada segmento toma la clase MAYORITARIA del prior fusionado (DW+CLC, el de mejor
     exactitud, 0,735) re-muestreado a 0,5 m por ventana.
  3. Píxeles sin segmento conservan la clase del prior; fuera de imagen -> IGNORE.

Reanudable por tesela: cada tesela se escribe como cover_{row}_{col}.tif y se registra en
un manifiesto; si el proceso muere (WSL reinicia), al relanzar salta las ya hechas.

Comandos:
  run    -> procesa teselas (reanudable)
  mosaic -> une teselas, recorta al AOI efectivo, vectoriza y homologa a destinos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v3.legend import IGNORE_INDEX, MODEL_LEGEND, model_to_destination_id  # noqa: E402
from pipeline_v3.objects_sam import build_mask_generator, masks_to_segment_ids, regularize_window  # noqa: E402

WS = ROOT / "workspace_rural_aoi"
MDT = WS / "data/prepared/mdt_2025_aoi_efectivo.tif"
TILE = 2048
# La mayoría por segmento solo se aplica a segmentos de hasta esta fracción de la
# tesela. Segmentos gigantes (SAM "rendido" en páramo homogéneo) conservan el prior
# píxel a píxel — continuo entre teselas — y así no hay flips temáticos por tesela.
MAX_SEG_FRAC = 0.20


def prior_window_aligned(prior, ortho, win: Window) -> np.ndarray:
    """Prior (10 m) re-muestreado (nearest) a la ventana de la orto (0,5 m).

    Usa la banda completa como fuente (``rasterio.band``) para que GDAL resuelva la
    grilla de re-muestreo de forma GLOBAL e idéntica en todas las teselas. (La versión
    anterior recortaba la ventana fuente con round_offsets(), lo que desplazaba la
    grilla hasta 5 m distinto por tesela → astillas/franjas en los bordes.)
    """
    dst = np.full((int(win.height), int(win.width)), IGNORE_INDEX, dtype=np.uint8)
    reproject(
        source=rasterio.band(prior, 1), destination=dst,
        dst_transform=ortho.window_transform(win), dst_crs=ortho.crs,
        src_nodata=IGNORE_INDEX, dst_nodata=IGNORE_INDEX, resampling=Resampling.nearest,
    )
    return dst


def _elev_window(mdt, ortho, win: Window) -> np.ndarray:
    """Elevación (MDT 10 m) re-muestreada (bilinear) a la ventana de la orto (0,5 m)."""
    dst = np.full((int(win.height), int(win.width)), 3200.0, dtype=np.float32)
    reproject(
        source=rasterio.band(mdt, 1), destination=dst,
        dst_transform=ortho.window_transform(win), dst_crs=ortho.crs,
        resampling=Resampling.bilinear,
    )
    return dst


def cmd_run(args) -> None:
    tiles_dir = args.out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.json"
    done = set(json.loads(manifest.read_text())) if manifest.exists() else set()

    with rasterio.open(args.ortho) as ortho:
        W, H = ortho.width, ortho.height
    positions = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    todo = [(r, c) for r, c in positions if f"{r}_{c}" not in done]
    print(f"teselas: {len(positions)} | hechas: {len(done)} | pendientes: {len(todo)}", flush=True)
    if not todo:
        print("nada pendiente.")
        return

    # modo de clasificación del segmento: 'prior' (voto DW) o 'rf' (Random Forest + GLCM)
    rf = None
    mdt_ds = None
    if args.classifier == "rf":
        import joblib
        rf = joblib.load(args.rf_model)["rf"]
        mdt_ds = rasterio.open(MDT)
        print(f"clasificador: RF+GLCM ({args.rf_model})", flush=True)
    else:
        print("clasificador: voto del prior DW", flush=True)

    sam = build_mask_generator(points_per_side=args.points_per_side,
                               min_mask_region_area=args.min_area)
    t0 = time.time()
    with rasterio.open(args.ortho) as ortho, rasterio.open(args.prior) as prior:
        for i, (r, c) in enumerate(todo):
            win = Window(c, r, min(TILE, W - c), min(TILE, H - r))
            ortho4 = ortho.read([1, 2, 3, 4], window=win)
            rgb = np.moveaxis(ortho4[:3], 0, -1).copy()
            prof = ortho.profile.copy()
            prof.update(height=int(win.height), width=int(win.width), count=1, dtype="uint8",
                        nodata=IGNORE_INDEX, transform=ortho.window_transform(win),
                        compress="lzw", tiled=True, blockxsize=256, blockysize=256)
            prof.pop("photometric", None)

            empty = not np.any(rgb)
            pri = prior_window_aligned(prior, ortho, win)
            if empty:
                cover = np.full((int(win.height), int(win.width)), IGNORE_INDEX, dtype=np.uint8)
            else:
                masks = sam.mask_generator.generate(rgb)
                max_px = int(MAX_SEG_FRAC * rgb.shape[0] * rgb.shape[1])
                masks = [m for m in masks if int(m["area"]) <= max_px]
                seg = masks_to_segment_ids(masks, rgb.shape[:2])
                if rf is not None:
                    from pipeline_v3.obia_rf import classify_segments
                    elev = _elev_window(mdt_ds, ortho, win)
                    cover = classify_segments(seg, ortho4, elev, pri, rf, ignore=IGNORE_INDEX)
                else:
                    cover = regularize_window(pri, seg, ignore=IGNORE_INDEX)
                cover[np.all(ortho4 == 0, axis=0)] = IGNORE_INDEX

            with rasterio.open(tiles_dir / f"cover_{r}_{c}.tif", "w", **prof) as dst:
                dst.write(cover, 1)
            done.add(f"{r}_{c}")
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                manifest.write_text(json.dumps(sorted(done)))
                el = time.time() - t0
                rate = el / (i + 1)
                eta = rate * (len(todo) - i - 1) / 3600
                print(f"  {len(done)}/{len(positions)} | {rate:.1f}s/tesela | ETA {eta:.1f} h", flush=True)
    print("teselas completas.")


def cmd_mosaic(args) -> None:
    from osgeo import gdal
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as to_shape

    tiles = sorted((args.out_dir / "tiles").glob("cover_*.tif"))
    if not tiles:
        raise SystemExit("no hay teselas; corre 'run' primero")
    print(f"mosaicando {len(tiles)} teselas...", flush=True)
    vrt = args.out_dir / "cover.vrt"
    gdal.BuildVRT(str(vrt), [str(t) for t in tiles])
    cover_tif = args.out_dir / "cobertura_2025_priorSAM.tif"
    gdal.Translate(str(cover_tif), str(vrt),
                   creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])

    # recorte exacto al AOI efectivo
    aoi = WS / "data/aoi/aoi_efectivo.gpkg"
    cover_clip = args.out_dir / "cobertura_2025_priorSAM_aoi.tif"
    gdal.Warp(str(cover_clip), str(cover_tif), cutlineDSName=str(aoi),
              cutlineLayer="aoi_efectivo", cropToCutline=True, dstNodata=IGNORE_INDEX,
              creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    print(f"cobertura -> {cover_clip}", flush=True)

    # sieve: elimina parches < ~0,1 ha (4000 px a 0,5 m) fusionándolos al vecino mayor
    print("sieve (parches < 0,1 ha)...", flush=True)
    ds = gdal.Open(str(cover_clip), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    gdal.SieveFilter(srcBand=band, maskBand=None, dstBand=band, threshold=4000,
                     connectedness=8)
    band.FlushCache()
    ds = None

    # vectorización + homologación a destinos
    print("vectorizando...", flush=True)
    polys, vals = [], []
    with rasterio.open(cover_clip) as src:
        for geom, v in rio_shapes(src.read(1), mask=src.read(1) != IGNORE_INDEX, transform=src.transform):
            polys.append(to_shape(geom)); vals.append(int(v))
        crs = src.crs
    gdf = gpd.GeoDataFrame({"model_id": vals}, geometry=polys, crs=crs)
    gdf["model_class"] = gdf["model_id"].map(lambda i: MODEL_LEGEND[i].name)
    m2d = model_to_destination_id()
    gdf["destino"] = gdf["model_id"].map(m2d)
    gdf["area_ha"] = gdf.geometry.area / 1e4
    gpkg = args.out_dir / "cobertura_2025_destinos.gpkg"
    gdf.to_file(gpkg, layer="cobertura", driver="GPKG")
    diss = gdf.dissolve(by="destino", as_index=False)[["destino", "geometry"]]
    diss.to_file(gpkg, layer="destinos", driver="GPKG")
    print(f"vector -> {gpkg} (capas cobertura, destinos)")
    print("\nárea por destino (ha):")
    print(gdf.groupby("destino")["area_ha"].sum().sort_values(ascending=False).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Producto final: prior + SAM 2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--ortho", type=Path, default=WS / "data/prepared/ortho_2025_aoi_efectivo.tif")
    pr.add_argument("--prior", type=Path, default=WS / "data/labels/weak_labels_fused.tif")
    pr.add_argument("--out-dir", type=Path, default=WS / "deliverables/producto_priorSAM")
    pr.add_argument("--points-per-side", type=int, default=24)
    pr.add_argument("--min-area", type=int, default=400)
    pr.add_argument("--classifier", choices=["prior", "rf"], default="prior")
    pr.add_argument("--rf-model", type=Path, default=WS / "models/obia_rf.joblib")
    pm = sub.add_parser("mosaic")
    pm.add_argument("--out-dir", type=Path, default=WS / "deliverables/producto_priorSAM")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "mosaic":
        cmd_mosaic(args)


if __name__ == "__main__":
    main()
