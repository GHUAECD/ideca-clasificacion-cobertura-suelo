"""Capa de objetos con SAM 2 y regularización por segmento (Fase 2).

Realiza la intuición "entrenar/entregar por objetos": SAM 2 delimita objetos de alta
calidad sobre la ortofoto 0.5 m (es *class-agnostic*: da la forma, no la clase) y la
clase la pone el modelo semántico (SegFormer/Clay). La predicción por píxel se
**regulariza por objeto**: cada segmento recibe la clase mayoritaria de los píxeles
que contiene → polígonos limpios, sin sal y pimienta, coherentes con cómo un
fotointérprete delimitaría.

Funciones:
  - ``segment_window``      SAM 2 automático sobre una ventana de la orto → ráster de
                            IDs de segmento georreferenciado (uint32, 0 = sin segmento).
  - ``regularize_window``   mayoría de clase del modelo por segmento (los píxeles sin
                            segmento conservan la clase del modelo).
  - CLI ``test-tile``       corre una tesela, reporta nº de segmentos y tiempo (para
                            dimensionar el costo del AOI completo).
  - CLI ``regularize``      aplica la regularización segmentos+predicción → salida.

La orto es R,G,B,NIR (bandas 1-4); SAM 2 usa sólo RGB. Ejecución en GPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

DEFAULT_TILE_PX = 2048  # 1.024 km a 0.5 m
DEFAULT_MODEL_ID = "sam2-hiera-large"


# ---------------------------------------------------------------------------
# SAM 2
# ---------------------------------------------------------------------------

def build_mask_generator(
    model_id: str = DEFAULT_MODEL_ID,
    points_per_side: int = 48,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.85,
    min_mask_region_area: int = 400,  # px; a 0.5 m son 100 m2
):
    """Crea el generador automático de máscaras de SAM 2 (descarga checkpoint si falta)."""
    import torch
    from samgeo.samgeo2 import SamGeo2

    sam = SamGeo2(
        model_id=model_id,
        automatic=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    return sam


def _read_rgb_window(ortho_path: Path, window: Window) -> tuple[np.ndarray, dict]:
    """Lee RGB (bandas 1-3) de una ventana como HxWx3 uint8 + perfil recortado."""
    with rasterio.open(ortho_path) as src:
        rgb = src.read([1, 2, 3], window=window)  # (3, h, w)
        profile = src.profile.copy()
        profile.update(
            height=int(window.height),
            width=int(window.width),
            transform=src.window_transform(window),
        )
    return np.moveaxis(rgb, 0, -1).copy(), profile


def masks_to_segment_ids(masks: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Convierte la lista de máscaras SAM (dicts con 'segmentation', 'area') a un ráster
    de IDs uint32. Máscaras más grandes primero, las pequeñas (más finas) pisan después,
    de modo que el detalle fino sobrevive. 0 = sin segmento."""
    seg = np.zeros(shape, dtype=np.uint32)
    for i, m in enumerate(sorted(masks, key=lambda m: -int(m["area"])), start=1):
        seg[m["segmentation"]] = i
    return seg


def segment_window(
    sam,
    ortho_path: Path,
    window: Window,
    out_path: Path | None = None,
) -> tuple[np.ndarray, dict, int]:
    """Segmenta una ventana de la orto con SAM 2. Devuelve (ids, perfil, n_segmentos)."""
    rgb, profile = _read_rgb_window(ortho_path, window)

    if not np.any(rgb):
        ids = np.zeros(rgb.shape[:2], dtype=np.uint32)
        n = 0
    else:
        sam.mask_generator  # asegura inicialización
        masks = sam.mask_generator.generate(rgb)
        ids = masks_to_segment_ids(masks, rgb.shape[:2])
        n = len(masks)

    if out_path is not None:
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="uint32", nodata=0, compress="lzw")
        out_profile.pop("photometric", None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(ids, 1)
    return ids, profile, n


# ---------------------------------------------------------------------------
# Regularización por objeto
# ---------------------------------------------------------------------------

def regularize_window(pred: np.ndarray, segment_ids: np.ndarray, ignore: int = 255) -> np.ndarray:
    """Asigna a cada segmento la clase mayoritaria de la predicción dentro de él.

    Vectorizado con bincount sobre pares (segmento, clase). Píxeles con segmento 0
    (sin objeto SAM) conservan la clase original del modelo.
    """
    out = pred.copy()
    valid = (segment_ids > 0) & (pred != ignore)
    if not valid.any():
        return out

    seg = segment_ids[valid].astype(np.int64)
    cls = pred[valid].astype(np.int64)
    n_cls = int(cls.max()) + 1

    # mapa compacto de ids de segmento presentes
    uniq, seg_compact = np.unique(seg, return_inverse=True)
    counts = np.bincount(seg_compact * n_cls + cls, minlength=len(uniq) * n_cls)
    majority = counts.reshape(len(uniq), n_cls).argmax(axis=1).astype(pred.dtype)

    lut = np.zeros(int(uniq.max()) + 1, dtype=pred.dtype)
    lut[uniq] = majority
    out[valid] = lut[segment_ids[valid]]
    return out


def regularize_raster(pred_path: Path, segments_path: Path, out_path: Path, ignore: int = 255) -> Path:
    """Regulariza un ráster de predicción completo contra un ráster de segmentos
    (mismas dimensiones/grilla), por bloques."""
    with rasterio.open(pred_path) as pred_src, rasterio.open(segments_path) as seg_src:
        if (pred_src.width, pred_src.height) != (seg_src.width, seg_src.height):
            raise ValueError("Predicción y segmentos no comparten dimensiones.")
        profile = pred_src.profile.copy()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, win in pred_src.block_windows(1):
                pred = pred_src.read(1, window=win)
                seg = seg_src.read(1, window=win)
                dst.write(regularize_window(pred, seg, ignore), 1, window=win)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_test_tile(args) -> None:
    with rasterio.open(args.ortho) as src:
        width, height = src.width, src.height
    tile = int(args.tile_px)
    col_off = int(args.col) if args.col is not None else (width - tile) // 2
    row_off = int(args.row) if args.row is not None else (height - tile) // 2
    window = Window(col_off, row_off, min(tile, width - col_off), min(tile, height - row_off))
    print(f"Tesela: off=({col_off},{row_off}) tam=({int(window.width)}x{int(window.height)}) px")

    t0 = time.perf_counter()
    sam = build_mask_generator(args.model_id, points_per_side=args.points_per_side)
    t_init = time.perf_counter() - t0
    print(f"Init SAM 2 ({args.model_id}): {t_init:.1f} s")

    t0 = time.perf_counter()
    out = Path(args.out) if args.out else None
    _, _, n = segment_window(sam, args.ortho, window, out)
    t_seg = time.perf_counter() - t0

    px = int(window.width) * int(window.height)
    total_tiles = (width // tile + 1) * (height // tile + 1)
    print(f"Segmentos: {n} | tiempo: {t_seg:.1f} s | {px / max(t_seg, 1e-9) / 1e6:.2f} Mpx/s")
    print(f"Proyección AOI completo (~{total_tiles} teselas de {tile}px): "
          f"~{total_tiles * t_seg / 3600:.1f} h en esta GPU")
    if out:
        print(f"Segmentos -> {out}")


def _cmd_regularize(args) -> None:
    out = regularize_raster(Path(args.pred), Path(args.segments), Path(args.out))
    print(f"Predicción regularizada por objeto -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Capa de objetos SAM 2 + regularización (Fase 2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("test-tile", help="Segmentar una tesela y medir tiempo")
    pt.add_argument("--ortho", required=True, type=Path)
    pt.add_argument("--tile-px", type=int, default=DEFAULT_TILE_PX)
    pt.add_argument("--row", type=int, default=None, help="row_off (default: centro)")
    pt.add_argument("--col", type=int, default=None, help="col_off (default: centro)")
    pt.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    pt.add_argument("--points-per-side", type=int, default=48)
    pt.add_argument("--out", type=Path, default=None)

    pr = sub.add_parser("regularize", help="Clase mayoritaria por segmento")
    pr.add_argument("--pred", required=True)
    pr.add_argument("--segments", required=True)
    pr.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "test-tile":
        _cmd_test_tile(args)
    elif args.cmd == "regularize":
        _cmd_regularize(args)


if __name__ == "__main__":
    main()
