"""Refinamiento dirigido: extraer el herbazal (liso) de dentro del Arbustal (ADENDA).

Sobre el producto DL fino (`producto_dl/cobertura_2025_dl_aoi.tif`), reclasifica a
`Pastos_y_herbaceo` (2) los píxeles que el SegFormer marcó como `Arbustal_y_secundaria`
(4) pero que son **lisos y poco verdes** — es decir, herbazal con arbustos dispersos que
DW/el modelo confundieron con arbustal. NINGUNA otra clase se toca: la única transición
posible es 4 → 2.

Criterio CONSERVADOR calibrado con la verdad puntual (Pastos vs Arbustal):
    cover == 4  AND  textura_std_local < UMBRAL_STD  AND  ndvi_local < UMBRAL_NDVI
con ventana ~16 m (32 px), igual a la usada en la calibración. Resultado en los puntos:
captura ~43 % del herbazal liso con ~11 % de arbustal real mal reclasificado.

Métricas rápidas vía filtros (uniform_filter), sin GLCM móvil (costoso). La orto se
reproyecta a la grilla exacta del cover por bloque, garantizando alineación píxel a píxel.

Uso:
    python -m pipeline_v3.refine_arbustal run        # genera el ráster refinado
    python -m pipeline_v3.refine_arbustal finalize   # sieve + vectoriza + homologa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_v3.legend import IGNORE_INDEX, MODEL_LEGEND, model_to_destination_id  # noqa: E402

WS = ROOT / "workspace_rural_aoi"
COVER = WS / "deliverables/producto_dl/cobertura_2025_dl_aoi.tif"
ORTHO = WS / "data/prepared/ortho_2025_aoi_efectivo.tif"
OUT_DIR = WS / "deliverables/producto_dl_refinado"

ARBUSTAL, PASTOS = 4, 2
TEX_WIN = 32          # ventana de textura/ndvi (~16 m), igual a la calibración (WIN_R=16)
UMBRAL_STD = 9.0      # textura por debajo => liso  (conservador)
UMBRAL_NDVI = 0.22    # ndvi por debajo => herbáceo, no leñoso
BLOCK = 2048
HALO = TEX_WIN        # halo para que el filtro no tenga costura de borde


def _ortho_on_grid(ortho_src, cover_src, win: Window) -> np.ndarray:
    """Lee RGBN de la orto reproyectada a la grilla EXACTA de la ventana del cover."""
    dst = np.zeros((4, int(win.height), int(win.width)), dtype=np.float32)
    for b in range(4):
        reproject(
            source=rasterio.band(ortho_src, b + 1), destination=dst[b],
            dst_transform=cover_src.window_transform(win), dst_crs=cover_src.crs,
            resampling=Resampling.bilinear,
        )
    return dst


def cmd_run(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tif = OUT_DIR / "cobertura_2025_dl_refinado_aoi.tif"
    with rasterio.open(COVER) as cov, rasterio.open(ORTHO) as ort:
        profile = cov.profile.copy()
        profile.update(compress="lzw", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="YES")
        W, H = cov.width, cov.height
        total_changed = 0
        with rasterio.open(out_tif, "w", **profile) as dst:
            for row in range(0, H, BLOCK):
                for col in range(0, W, BLOCK):
                    ch = min(BLOCK, H - row)
                    cw = min(BLOCK, W - col)
                    # leer con halo
                    r0 = max(0, row - HALO); c0 = max(0, col - HALO)
                    r1 = min(H, row + ch + HALO); c1 = min(W, col + cw + HALO)
                    win = Window(c0, r0, c1 - c0, r1 - r0)
                    cover = cov.read(1, window=win)
                    if not np.any(cover == ARBUSTAL):
                        dst.write(cover[row - r0:row - r0 + ch, col - c0:col - c0 + cw], 1,
                                  window=Window(col, row, cw, ch))
                        continue
                    o4 = _ortho_on_grid(ort, cov, win)
                    r_, g_, b_, nir = o4
                    brillo = (r_ + g_ + b_) / 3.0
                    ndvi = (nir - r_) / np.clip(nir + r_, 1, None)
                    # textura local = std del brillo en ventana TEX_WIN
                    mean_b = uniform_filter(brillo, TEX_WIN, mode="reflect")
                    mean_b2 = uniform_filter(brillo * brillo, TEX_WIN, mode="reflect")
                    std_local = np.sqrt(np.clip(mean_b2 - mean_b * mean_b, 0, None))
                    ndvi_local = uniform_filter(ndvi, TEX_WIN, mode="reflect")

                    liso = (cover == ARBUSTAL) & (std_local < UMBRAL_STD) & (ndvi_local < UMBRAL_NDVI)
                    out = cover.copy()
                    out[liso] = PASTOS
                    # recortar el core (quitar halo) y escribir
                    cr0 = row - r0; cc0 = col - c0
                    core = out[cr0:cr0 + ch, cc0:cc0 + cw]
                    total_changed += int(liso[cr0:cr0 + ch, cc0:cc0 + cw].sum())
                    dst.write(core, 1, window=Window(col, row, cw, ch))
                print(f"  fila {row}/{H} | reclasificado acumulado: {total_changed * 0.25 / 1e4:.0f} ha", flush=True)
    print(f"Arbustal->Herbazal reclasificado: {total_changed * 0.25 / 1e4:.1f} ha")
    print(f"ráster refinado -> {out_tif}")


def cmd_finalize(args) -> None:
    from osgeo import gdal
    import geopandas as gpd
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as to_shape

    cover_tif = OUT_DIR / "cobertura_2025_dl_refinado_aoi.tif"
    if args.sieve > 0:
        print(f"sieve ligero (< {args.sieve} px)...", flush=True)
        ds = gdal.Open(str(cover_tif), gdal.GA_Update)
        b = ds.GetRasterBand(1)
        gdal.SieveFilter(srcBand=b, maskBand=None, dstBand=b, threshold=args.sieve, connectedness=8)
        b.FlushCache(); ds = None

    print("vectorizando...", flush=True)
    polys, vals = [], []
    with rasterio.open(cover_tif) as src:
        for geom, v in rio_shapes(src.read(1), mask=src.read(1) != IGNORE_INDEX, transform=src.transform):
            polys.append(to_shape(geom)); vals.append(int(v))
        crs = src.crs
    gdf = gpd.GeoDataFrame({"model_id": vals}, geometry=polys, crs=crs)
    gdf["model_class"] = gdf["model_id"].map(lambda i: MODEL_LEGEND[i].name)
    gdf["destino"] = gdf["model_id"].map(model_to_destination_id())
    gdf["area_ha"] = gdf.geometry.area / 1e4
    gpkg = OUT_DIR / "cobertura_2025_dl_refinado_destinos.gpkg"
    gdf.to_file(gpkg, layer="cobertura", driver="GPKG")
    gdf.dissolve(by="destino", as_index=False)[["destino", "geometry"]].to_file(gpkg, layer="destinos", driver="GPKG")
    print(f"vector -> {gpkg}")
    print("\nárea por destino (ha):")
    print(gdf.groupby("destino")["area_ha"].sum().sort_values(ascending=False).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Refinamiento Arbustal->Herbazal (liso)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    pf = sub.add_parser("finalize")
    pf.add_argument("--sieve", type=int, default=1000)
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "finalize":
        cmd_finalize(args)


if __name__ == "__main__":
    main()
