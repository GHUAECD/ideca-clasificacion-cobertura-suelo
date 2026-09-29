"""Producto deep learning: mapa de cobertura 2025 con SegFormer v3 (grano fino 0,5 m).

A diferencia del producto grueso (prior DW + SAM), aquí la clase de cada píxel la
predice el SegFormer directamente sobre la orto 0,5 m, capturando la variación temática
fina (mosaicos, transiciones) que DW a 10 m promedia y borra.

Inferencia por teselas con *halo* (contexto extra) y mezcla por peso coseno entre chips
solapados → mapa continuo sin costuras. Reanudable por tesela (WSL se reinicia).

Comandos:
  run    -> infiere por teselas (reanudable) -> tiles/pred_{r}_{c}.tif
  mosaic -> une, recorta al AOI, (sieve ligero opcional) vectoriza y homologa a destinos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v2.utils import align_single_band, cosine_weight, feature_stack  # noqa: E402
from pipeline_v3.legend import IGNORE_INDEX, MODEL_LEGEND, model_to_destination_id, num_model_classes  # noqa: E402
from pipeline_v3.train_segformer import FEATURES, build_model  # noqa: E402

WS = ROOT / "workspace_rural_aoi"
ORTHO = WS / "data/prepared/ortho_2025_aoi_efectivo.tif"
MDT = WS / "data/prepared/mdt_2025_aoi_efectivo.tif"
TILE = 2048
HALO = 128
CHIP = 512
STRIDE = 256


def cmd_run(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = num_model_classes()
    model = build_model(args.hf_model, n)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"SegFormer v3: {args.checkpoint} (epoch {ckpt.get('epoch')})", flush=True)

    tiles_dir = args.out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.json"
    done = set(json.loads(manifest.read_text())) if manifest.exists() else set()
    weight = cosine_weight(CHIP)

    with rasterio.open(ORTHO) as ortho:
        W, H = ortho.width, ortho.height
    positions = [(r, c) for r in range(0, H, TILE) for c in range(0, W, TILE)]
    todo = [(r, c) for r, c in positions if f"{r}_{c}" not in done]
    print(f"teselas: {len(positions)} | hechas: {len(done)} | pendientes: {len(todo)}", flush=True)
    if not todo:
        print("nada pendiente."); return

    t0 = time.time()
    with rasterio.open(ORTHO) as ortho_src, rasterio.open(MDT) as mdt_src:
        for i, (row_off, col_off) in enumerate(todo):
            core_h = min(TILE, H - row_off)
            core_w = min(TILE, W - col_off)
            read_row = max(0, row_off - HALO)
            read_col = max(0, col_off - HALO)
            read_h = min(H - read_row, core_h + (row_off - read_row) + HALO)
            read_w = min(W - read_col, core_w + (col_off - read_col) + HALO)
            win = Window(read_col, read_row, read_w, read_h)
            ortho4 = ortho_src.read([1, 2, 3, 4], window=win)

            prof = ortho_src.profile.copy()
            prof.update(count=1, dtype="uint8", nodata=IGNORE_INDEX,
                        height=core_h, width=core_w,
                        transform=ortho_src.window_transform(Window(col_off, row_off, core_w, core_h)),
                        compress="lzw", tiled=True, blockxsize=256, blockysize=256)
            prof.pop("photometric", None)

            if not np.any(ortho4):
                cover = np.full((core_h, core_w), IGNORE_INDEX, dtype=np.uint8)
            else:
                elev = align_single_band(mdt_src, ortho_src, win)
                stack = feature_stack(ortho4, elev, pixel_size=float(ortho_src.res[0]),
                                      elevation_min=FEATURES["elevation_min"],
                                      elevation_max=FEATURES["elevation_max"],
                                      slope_scale=FEATURES["slope_scale"])
                h, w = stack.shape[-2:]
                logits_sum = np.zeros((n, h, w), dtype=np.float32)
                wsum = np.zeros((h, w), dtype=np.float32)
                jobs = [(r, c) for r in _positions(h, CHIP, STRIDE) for c in _positions(w, CHIP, STRIDE)]
                for s in range(0, len(jobs), args.batch_size):
                    bj = jobs[s:s + args.batch_size]
                    batch = np.stack([stack[:, r:r + CHIP, c:c + CHIP] for r, c in bj])
                    with torch.no_grad(), torch.amp.autocast(device):
                        lo = model(pixel_values=torch.from_numpy(batch).to(device)).logits
                        lo = F.interpolate(lo, size=(CHIP, CHIP), mode="bilinear", align_corners=False)
                    lo = lo.float().cpu().numpy()
                    for k, (r, c) in enumerate(bj):
                        logits_sum[:, r:r + CHIP, c:c + CHIP] += lo[k] * weight
                        wsum[r:r + CHIP, c:c + CHIP] += weight
                pred = np.argmax(logits_sum / np.clip(wsum, 1e-6, None), axis=0).astype(np.uint8)
                # recorta el core (quita el halo)
                cr0 = row_off - read_row
                cc0 = col_off - read_col
                pred = pred[cr0:cr0 + core_h, cc0:cc0 + core_w]
                core4 = ortho4[:, cr0:cr0 + core_h, cc0:cc0 + core_w]
                pred[np.all(core4 == 0, axis=0)] = IGNORE_INDEX
                cover = pred

            with rasterio.open(tiles_dir / f"pred_{row_off}_{col_off}.tif", "w", **prof) as dst:
                dst.write(cover, 1)
            done.add(f"{row_off}_{col_off}")
            if (i + 1) % 5 == 0 or i == len(todo) - 1:
                manifest.write_text(json.dumps(sorted(done)))
                rate = (time.time() - t0) / (i + 1)
                print(f"  {len(done)}/{len(positions)} | {rate:.1f}s/tesela | "
                      f"ETA {rate * (len(todo) - i - 1) / 3600:.1f} h", flush=True)
    print("teselas completas.")


def _positions(length, chip, stride):
    if length <= chip:
        return [0]
    pos = list(range(0, length - chip + 1, stride))
    if pos[-1] != length - chip:
        pos.append(length - chip)
    return pos


def cmd_mosaic(args) -> None:
    from osgeo import gdal
    import geopandas as gpd
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as to_shape

    tiles = sorted((args.out_dir / "tiles").glob("pred_*.tif"))
    print(f"mosaicando {len(tiles)} teselas...", flush=True)
    vrt = args.out_dir / "pred.vrt"
    gdal.BuildVRT(str(vrt), [str(t) for t in tiles])
    full = args.out_dir / "cobertura_2025_dl.tif"
    gdal.Translate(str(full), str(vrt), creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    clip = args.out_dir / "cobertura_2025_dl_aoi.tif"
    gdal.Warp(str(clip), str(full), cutlineDSName=str(WS / "data/aoi/aoi_efectivo.gpkg"),
              cutlineLayer="aoi_efectivo", cropToCutline=True, dstNodata=IGNORE_INDEX,
              creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    if args.sieve > 0:
        print(f"sieve ligero (< {args.sieve} px)...", flush=True)
        ds = gdal.Open(str(clip), gdal.GA_Update)
        b = ds.GetRasterBand(1)
        gdal.SieveFilter(srcBand=b, maskBand=None, dstBand=b, threshold=args.sieve, connectedness=8)
        b.FlushCache(); ds = None
    print(f"cobertura DL -> {clip}", flush=True)

    print("vectorizando...", flush=True)
    polys, vals = [], []
    with rasterio.open(clip) as src:
        for geom, v in rio_shapes(src.read(1), mask=src.read(1) != IGNORE_INDEX, transform=src.transform):
            polys.append(to_shape(geom)); vals.append(int(v))
        crs = src.crs
    gdf = gpd.GeoDataFrame({"model_id": vals}, geometry=polys, crs=crs)
    gdf["model_class"] = gdf["model_id"].map(lambda i: MODEL_LEGEND[i].name)
    gdf["destino"] = gdf["model_id"].map(model_to_destination_id())
    gdf["area_ha"] = gdf.geometry.area / 1e4
    gpkg = args.out_dir / "cobertura_2025_dl_destinos.gpkg"
    gdf.to_file(gpkg, layer="cobertura", driver="GPKG")
    gdf.dissolve(by="destino", as_index=False)[["destino", "geometry"]].to_file(gpkg, layer="destinos", driver="GPKG")
    print(f"vector -> {gpkg}")
    print("\nárea por destino (ha):")
    print(gdf.groupby("destino")["area_ha"].sum().sort_values(ascending=False).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Producto DL: SegFormer v3 fino")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--checkpoint", type=Path, default=WS / "models/segformer_v3/best.pt")
    pr.add_argument("--hf-model", default="nvidia/mit-b2")
    pr.add_argument("--out-dir", type=Path, default=WS / "deliverables/producto_dl")
    pr.add_argument("--batch-size", type=int, default=8)
    pm = sub.add_parser("mosaic")
    pm.add_argument("--out-dir", type=Path, default=WS / "deliverables/producto_dl")
    pm.add_argument("--sieve", type=int, default=1000, help="px mínimos (1000≈0,025 ha); 0 = sin sieve")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "mosaic":
        cmd_mosaic(args)


if __name__ == "__main__":
    main()
