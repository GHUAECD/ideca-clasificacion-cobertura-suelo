"""Validación del Clay+LoRA contra los puntos de verdad 2025 (misma vara que el resto).

Replica cmd_points de validate_2025.py pero con el ClaySegmenter: ventana de 512 px
centrada en el punto -> submuestreo a 256 (igual que en entrenamiento) -> predicción
-> reescalado a 512 -> clase modal en parche 5x5 alrededor del punto.
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

from pipeline_v3.legend import MODEL_LEGEND, destination_of, num_model_classes  # noqa: E402
from pipeline_v3.train_clay_lora import INPUT_PX, ClaySegmenter  # noqa: E402
from pipeline_v3.validate_2025 import _metrics_from_cm, centered_window  # noqa: E402

WS = ROOT / "workspace_rural_aoi"
CHIP = 512
PATCH = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=WS / "models/clay_lora/best.pt")
    ap.add_argument("--out", type=Path, default=WS / "deliverables/validation_2025_clay.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = num_model_classes()
    model = ClaySegmenter(n)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    print(f"checkpoint: {args.checkpoint} (epoch {state.get('epoch')}, f1_weak {state.get('val_macro_f1_weak', 0):.3f})")

    name_to_id = {mc.name: cid for cid, mc in MODEL_LEGEND.items()}
    pts = gpd.read_file(WS / "deliverables/val_set_2025.gpkg", layer="val_set_2025")
    extra = gpd.read_file(WS / "deliverables/val_set_2025_complemento.gpkg", layer="val_set_complemento").to_crs(pts.crs)
    pts = gpd.GeoDataFrame(pd.concat([pts, extra], ignore_index=True), crs=pts.crs)
    aoi = gpd.read_file(WS / "data/aoi/aoi_efectivo.gpkg", layer="aoi_efectivo")
    pts = pts.to_crs(aoi.crs)
    pts = pts[pts.within(aoi.union_all())]
    pts = pts[pts["model_class"].fillna("").str.strip().isin(name_to_id)].reset_index(drop=True)
    minx, miny, maxx, maxy = aoi.total_bounds
    h = maxy - miny
    t1, t2 = maxy - h / 3, maxy - 2 * h / 3
    pts["zona"] = np.where(pts.geometry.y > t1, "norte", np.where(pts.geometry.y > t2, "centro", "sur"))
    print(f"puntos evaluables: {len(pts)}")

    cm9 = np.zeros((n, n), dtype=np.int64)
    rows = []
    with rasterio.open(WS / "data/prepared/ortho_2025_aoi_efectivo.tif") as ortho:
        pts = pts.to_crs(ortho.crs)
        for _, p in pts.iterrows():
            win = centered_window(ortho, p.geometry.x, p.geometry.y, CHIP)
            if win is None:
                continue
            bgrn = ortho.read([3, 2, 1, 4], window=win).astype(np.float32) / 255.0
            x = torch.from_numpy(bgrn)[None]
            x = F.interpolate(x, size=(INPUT_PX, INPUT_PX), mode="bilinear", align_corners=False)
            with torch.no_grad(), torch.amp.autocast(device):
                logits = model(x.to(device))
            logits = F.interpolate(logits, size=(CHIP, CHIP), mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            row, col = ortho.index(p.geometry.x, p.geometry.y)
            pr, pc = int(row - win.row_off), int(col - win.col_off)
            half = PATCH // 2
            patch = pred[max(0, pr - half): pr + half + 1, max(0, pc - half): pc + half + 1]
            pred_id = int(Counter(patch.ravel().tolist()).most_common(1)[0][0])
            true_id = name_to_id[p["model_class"].strip()]
            cm9[true_id, pred_id] += 1
            rows.append({"sample_id": p["sample_id"], "zona": p["zona"], "true": true_id, "pred": pred_id})

    names9 = [MODEL_LEGEND[i].name for i in range(n)]
    res9 = _metrics_from_cm(cm9, names9)
    dests = sorted({mc.destination for mc in MODEL_LEGEND.values()})
    di = {d: i for i, d in enumerate(dests)}
    cmd_ = np.zeros((len(dests), len(dests)), dtype=np.int64)
    for t in range(n):
        for q in range(n):
            cmd_[di[destination_of(t)], di[destination_of(q)]] += cm9[t, q]
    resd = _metrics_from_cm(cmd_, dests)

    detail = pd.DataFrame(rows)
    by_zone = {}
    for zona, grp in detail.groupby("zona"):
        td = grp["true"].map(lambda i: destination_of(int(i)))
        pdd = grp["pred"].map(lambda i: destination_of(int(i)))
        by_zone[zona] = {"n": int(len(grp)),
                         "acc_9clases": round(float((grp["true"] == grp["pred"]).mean()), 3),
                         "acc_destinos": round(float((td == pdd).mean()), 3)}

    print(f"\n== CLAY 9 clases ==  acc {res9['accuracy']:.3f} | macroF1 {res9['macro_f1_present']:.3f} | n {res9['n']}")
    print(f"== CLAY 6 destinos == acc {resd['accuracy']:.3f} | macroF1 {resd['macro_f1_present']:.3f}")
    for d, m in resd["per_class"].items():
        print(f"  {d:22s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} (n={m['support']})")
    print("== por zona ==")
    for zona, m in by_zone.items():
        print(f"  {zona:8s} n={m['n']:4d} | 9c {m['acc_9clases']:.3f} | destinos {m['acc_destinos']:.3f}")

    out = {"points_9class": res9, "points_destinations": resd, "by_zone": by_zone}
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    detail.to_csv(WS / "deliverables/val_points_detail_clay.csv", index=False)
    print(f"\nresultados -> {args.out}")


if __name__ == "__main__":
    main()
