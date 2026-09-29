"""Validación contra la verdad 2025 (Fase 4 — la métrica que importa).

Dos evaluaciones complementarias:

1. ``points``  — exactitud puntual contra el set etiquetado por el usuario
   (``val_set_2025.gpkg``): para cada punto con ``model_class``, se predice una
   ventana de 512 px centrada y se toma la clase modal en un parche de 5x5 px
   (~2,5 m) alrededor del punto. Reporta matriz de confusión, accuracy y F1 por
   clase (9) y colapsado a destinos económicos (6). Este es el número honesto.

2. ``vicinity`` — chequeo de vecindad contra los puntos de campo fotográficos
   (``field_usable``): el punto está sobre la vía, así que se exige que la clase
   declarada exista en la predicción dentro de un radio (~75 m). Tasa de acuerdo
   por clase. Es una métrica de plausibilidad, no de exactitud puntual.

Uso:
    python -m pipeline_v3.validate_2025 points --checkpoint .../best.pt
    python -m pipeline_v3.validate_2025 vicinity --checkpoint .../best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v2.utils import align_single_band, feature_stack  # noqa: E402
from pipeline_v3.legend import (  # noqa: E402
    IGNORE_INDEX,
    MODEL_LEGEND,
    destination_of,
    num_model_classes,
)
from pipeline_v3.train_segformer import FEATURES, IN_CHANNELS, build_model  # noqa: E402

WS = ROOT / "workspace_rural_aoi"
DEFAULTS = {
    "ortho": WS / "data/prepared/ortho_2025_aoi_efectivo.tif",
    "mdt": WS / "data/prepared/mdt_2025_aoi_efectivo.tif",
    "val_set": WS / "deliverables/val_set_2025.gpkg",
    "field": WS / "data/field/puntos_campo_2026.gpkg",
    "aoi": WS / "data/aoi/aoi_efectivo.gpkg",
}
CHIP = 512
PATCH = 5          # parche modal alrededor del punto (px)
VICINITY_M = 75.0  # radio del chequeo de vecindad


def load_model(checkpoint: Path, hf_model: str, device: str):
    model = build_model(hf_model, num_model_classes())
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    print(f"checkpoint: {checkpoint} (epoch {state.get('epoch')}, "
          f"val_f1_weak {state.get('val_macro_f1_weak', float('nan')):.3f})")
    return model


def predict_window(model, ortho, mdt, win: Window, device: str) -> np.ndarray:
    x = feature_stack(
        ortho.read([1, 2, 3, 4], window=win),
        align_single_band(mdt, ortho, win),
        pixel_size=float(ortho.res[0]),
        elevation_min=FEATURES["elevation_min"],
        elevation_max=FEATURES["elevation_max"],
        slope_scale=FEATURES["slope_scale"],
    )
    with torch.no_grad(), torch.amp.autocast(device):
        t = torch.from_numpy(x[None]).to(device)
        logits = model(pixel_values=t).logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def centered_window(ortho, x: float, y: float, size: int) -> Window | None:
    row, col = ortho.index(x, y)
    r0, c0 = row - size // 2, col - size // 2
    if r0 < 0 or c0 < 0 or r0 + size > ortho.height or c0 + size > ortho.width:
        r0 = min(max(r0, 0), ortho.height - size)
        c0 = min(max(c0, 0), ortho.width - size)
        if r0 < 0 or c0 < 0:
            return None
    return Window(c0, r0, size, size)


def _metrics_from_cm(cm: np.ndarray, names: list[str]) -> dict:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1)
    precision = tp / np.maximum(cm.sum(0), 1e-9)
    recall = tp / np.maximum(support, 1e-9)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    present = support > 0
    return {
        "n": int(cm.sum()),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1_present": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            names[i]: {"precision": round(float(precision[i]), 3),
                       "recall": round(float(recall[i]), 3),
                       "f1": round(float(f1[i]), 3),
                       "support": int(support[i])}
            for i in range(len(names)) if support[i] > 0 or cm.sum(0)[i] > 0
        },
        "confusion_matrix": cm.tolist(),
    }


def cmd_points(args, model, device) -> dict:
    name_to_id = {mc.name: cid for cid, mc in MODEL_LEGEND.items()}
    n = num_model_classes()

    pts = gpd.read_file(args.val_set, layer="val_set_2025")
    if args.val_extra and Path(args.val_extra).exists():
        extra = gpd.read_file(args.val_extra, layer="val_set_complemento").to_crs(pts.crs)
        pts = gpd.GeoDataFrame(pd.concat([pts, extra], ignore_index=True), crs=pts.crs)
        print(f"set complementario incluido: {len(extra)} puntos")
    aoi = gpd.read_file(args.aoi, layer="aoi_efectivo")
    pts = pts.to_crs(aoi.crs)
    pts = pts[pts.within(aoi.union_all())]
    pts = pts[pts["model_class"].fillna("").str.strip().isin(name_to_id)].reset_index(drop=True)
    print(f"puntos evaluables (etiquetados y dentro del AOI): {len(pts)}")

    # zona por tercio N-S del AOI (para detectar sesgo espacial del desempeño)
    minx, miny, maxx, maxy = aoi.total_bounds
    h = maxy - miny
    t1, t2 = maxy - h / 3, maxy - 2 * h / 3
    pts["zona"] = np.where(pts.geometry.y > t1, "norte",
                   np.where(pts.geometry.y > t2, "centro", "sur"))

    cm9 = np.zeros((n, n), dtype=np.int64)
    rows = []
    with rasterio.open(args.ortho) as ortho, rasterio.open(args.mdt) as mdt:
        pts = pts.to_crs(ortho.crs)
        for _, p in pts.iterrows():
            win = centered_window(ortho, p.geometry.x, p.geometry.y, CHIP)
            if win is None:
                continue
            pred = predict_window(model, ortho, mdt, win, device)
            row, col = ortho.index(p.geometry.x, p.geometry.y)
            pr, pc = int(row - win.row_off), int(col - win.col_off)
            half = PATCH // 2
            patch = pred[max(0, pr - half): pr + half + 1, max(0, pc - half): pc + half + 1]
            pred_id = int(Counter(patch.ravel().tolist()).most_common(1)[0][0])
            true_id = name_to_id[p["model_class"].strip()]
            cm9[true_id, pred_id] += 1
            rows.append({"sample_id": p["sample_id"], "zona": p["zona"],
                         "true": true_id, "pred": pred_id})

    names9 = [MODEL_LEGEND[i].name for i in range(n)]
    res9 = _metrics_from_cm(cm9, names9)

    # colapso a destinos
    dests = sorted({mc.destination for mc in MODEL_LEGEND.values()})
    di = {d: i for i, d in enumerate(dests)}
    cmd_ = np.zeros((len(dests), len(dests)), dtype=np.int64)
    for t in range(n):
        for q in range(n):
            cmd_[di[destination_of(t)], di[destination_of(q)]] += cm9[t, q]
    resd = _metrics_from_cm(cmd_, dests)

    # métricas por zona N-S (sesgo espacial)
    detail = pd.DataFrame(rows)
    by_zone = {}
    for zona, grp in detail.groupby("zona"):
        acc9 = float((grp["true"] == grp["pred"]).mean())
        td = grp["true"].map(lambda i: destination_of(int(i)))
        pdest = grp["pred"].map(lambda i: destination_of(int(i)))
        by_zone[zona] = {"n": int(len(grp)), "acc_9clases": round(acc9, 3),
                         "acc_destinos": round(float((td == pdest).mean()), 3)}

    out = {"points_9class": res9, "points_destinations": resd, "by_zone": by_zone}
    print(f"\n== 9 clases ==  acc {res9['accuracy']:.3f} | macroF1 {res9['macro_f1_present']:.3f} | n {res9['n']}")
    print(f"== 6 destinos == acc {resd['accuracy']:.3f} | macroF1 {resd['macro_f1_present']:.3f}")
    for d, m in resd["per_class"].items():
        print(f"  {d:22s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} (n={m['support']})")
    print("== por zona N-S ==")
    for zona, m in by_zone.items():
        print(f"  {zona:8s} n={m['n']:4d} | acc 9c {m['acc_9clases']:.3f} | acc destinos {m['acc_destinos']:.3f}")
    detail.to_csv(args.out_dir / "val_points_detail.csv", index=False)
    return out


def cmd_vicinity(args, model, device) -> dict:
    pts = gpd.read_file(args.field, layer="field_usable")
    rad_px = int(VICINITY_M / 0.5)
    hits, total = Counter(), Counter()
    with rasterio.open(args.ortho) as ortho, rasterio.open(args.mdt) as mdt:
        pts = pts.to_crs(ortho.crs)
        for _, p in pts.iterrows():
            win = centered_window(ortho, p.geometry.x, p.geometry.y, CHIP)
            if win is None:
                continue
            pred = predict_window(model, ortho, mdt, win, device)
            row, col = ortho.index(p.geometry.x, p.geometry.y)
            pr, pc = int(row - win.row_off), int(col - win.col_off)
            r0, r1 = max(0, pr - rad_px), min(CHIP, pr + rad_px)
            c0, c1 = max(0, pc - rad_px), min(CHIP, pc + rad_px)
            neighborhood = pred[r0:r1, c0:c1]
            cls = p["model_class"]
            total[cls] += 1
            if (neighborhood == int(p["model_id"])).any():
                hits[cls] += 1

    res = {cls: {"agree": hits[cls], "n": total[cls], "rate": round(hits[cls] / total[cls], 3)}
           for cls in sorted(total)}
    overall = sum(hits.values()) / max(sum(total.values()), 1)
    print(f"\n== vecindad (campo, r={VICINITY_M:.0f} m) == acuerdo global {overall:.3f}")
    for cls, m in res.items():
        print(f"  {cls:22s} {m['agree']}/{m['n']} = {m['rate']:.2f}")
    return {"vicinity": res, "vicinity_overall": overall}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validación contra verdad 2025")
    ap.add_argument("cmd", choices=["points", "vicinity", "all"])
    ap.add_argument("--checkpoint", type=Path, default=WS / "models/segformer_v3/best.pt")
    ap.add_argument("--hf-model", default="nvidia/mit-b2")
    ap.add_argument("--val-extra", type=Path,
                    default=WS / "deliverables/val_set_2025_complemento.gpkg",
                    help="Set complementario centro/sur (se ignora si no existe)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=Path, default=v)
    ap.add_argument("--out-dir", type=Path, default=WS / "deliverables")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, args.hf_model, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    if args.cmd in ("points", "all"):
        out.update(cmd_points(args, model, device))
    if args.cmd in ("vicinity", "all"):
        out.update(cmd_vicinity(args, model, device))
    path = args.out_dir / "validation_2025.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultados -> {path}")


if __name__ == "__main__":
    main()
