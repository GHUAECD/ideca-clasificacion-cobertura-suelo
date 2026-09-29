"""Active learning — Hito 1: inyección de los puntos de campo al entrenamiento.

Los 351 puntos de campo (``field_usable``) son GPS sobre la VÍA, con la cobertura
declarada alrededor. No son verdad puntual exacta (no se puede pintar el píxel del
punto, que es carretera), pero sí son evidencia fuerte de QUÉ cobertura hay cerca —
y se concentran en Cultivos/Pastos, justo la frontera donde el prior débil falla.

Estrategia de inyección (conservadora, anti-contaminación):
  - Para cada punto, un **anillo** entre R_MIN (salta la vía/construido del propio punto)
    y R_MAX alrededor.
  - Dentro del anillo, se sobre-escribe la etiqueta débil con la clase de campo SOLO
    en píxeles cuyo prior ya es una clase herbácea/confundible
    {Cultivos, Pastos_y_herbaceo, Arbustal_y_secundaria}. Así se afina la frontera
    Cultivos↔Pastos sin pintar vías, agua, bosque, roca, páramo ni humedales.
  - Se excluyen puntos a < EXCL_VAL_M de cualquier punto de validación del usuario,
    para no filtrar verdad al conjunto de evaluación.

Salida: ``weak_labels_fused_field.tif`` (igual que el fusionado, pero con los anillos
de campo) + máscara ``field_chip_mask`` para sobre-muestrear esos chips en el entrenamiento.

Uso:
    python -m pipeline_v3.active_learning inject-field \
        --weak workspace_rural_aoi/data/labels/weak_labels_fused.tif \
        --field workspace_rural_aoi/data/field/puntos_campo_2026.gpkg \
        --val workspace_rural_aoi/deliverables/val_set_2025.gpkg \
        --val-extra workspace_rural_aoi/deliverables/val_set_2025_complemento.gpkg \
        --out workspace_rural_aoi/data/labels/weak_labels_fused_field.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from .legend import MODEL_LEGEND, IGNORE_INDEX

R_MIN_M = 12.0   # radio interno: salta la vía/punto
R_MAX_M = 40.0   # radio externo: cobertura vecina declarada
EXCL_VAL_M = 30.0  # excluye puntos cerca de validación (anti-fuga)
# Solo se corrige sobre estas clases del prior (herbáceas/confundibles)
OVERRIDABLE = {1, 2, 4}  # Cultivos, Pastos_y_herbaceo, Arbustal_y_secundaria


def inject_field(weak_path: Path, field_path: Path, val_paths: list[Path], out_path: Path) -> Path:
    with rasterio.open(weak_path) as src:
        profile = src.profile.copy()
        labels = src.read(1)
        transform = src.transform
        crs = src.crs
        res = abs(src.res[0])

    field = gpd.read_file(field_path, layer="field_usable").to_crs(crs)

    # geometrías de validación a excluir
    val_geoms = []
    for vp in val_paths:
        if vp and Path(vp).exists():
            layer = "val_set_complemento" if "complemento" in vp.name else "val_set_2025"
            try:
                g = gpd.read_file(vp, layer=layer).to_crs(crs)
            except Exception:
                g = gpd.read_file(vp).to_crs(crs)
            val_geoms.append(g)
    val_union = None
    if val_geoms:
        allv = gpd.GeoDataFrame(__import__("pandas").concat(val_geoms, ignore_index=True), crs=crs)
        val_union = allv.buffer(EXCL_VAL_M).union_all()

    out = labels.copy()
    n_used = 0
    changed_total = 0
    overridable_mask_cache = np.isin(labels, list(OVERRIDABLE))

    for _, p in field.iterrows():
        geom = p.geometry
        if val_union is not None and geom.within(val_union):
            continue
        cls = int(p["model_id"])
        if cls < 0:
            continue
        ring = geom.buffer(R_MAX_M).difference(geom.buffer(R_MIN_M))
        if ring.is_empty:
            continue
        ring_mask = rasterize([(ring, 1)], out_shape=labels.shape, transform=transform,
                              fill=0, dtype=np.uint8).astype(bool)
        target = ring_mask & overridable_mask_cache
        if not target.any():
            continue
        out[target] = cls
        changed_total += int(target.sum())
        n_used += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

    print(f"Inyección de campo -> {out_path}")
    print(f"  puntos usados (fuera de validación, con prior corregible): {n_used}/{len(field)}")
    print(f"  píxeles re-etiquetados: {changed_total:,}")
    # distribución de clases de campo aplicadas
    diff = out != labels
    if diff.any():
        vals, counts = np.unique(out[diff], return_counts=True)
        print("  por clase de campo aplicada:")
        for v, c in sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: -kv[1]):
            print(f"    {MODEL_LEGEND[int(v)].name:22s} {c:>10,}")
    return out_path


AL_DISC_M = 15.0  # radio del disco por punto AL; >=10m para registrar en la grilla de 10m
                  # del label (cada celda 10m -> 20x20 px a 0.5m al re-muestrear en entrenamiento)


def inject_active_points(weak_path: Path, al_path: Path, layer: str, out_path: Path) -> Path:
    """Estampa discos de verdad EXACTA de los puntos de active learning etiquetados.

    A diferencia de los puntos de campo (GPS sobre la vía, con desfase), estos los
    etiquetó el usuario sobre la ubicación precisa en la orto, así que sobre-escriben
    la etiqueta débil sin restricción de clase, con un disco pequeño (R=AL_DISC_M).
    """
    with rasterio.open(weak_path) as src:
        profile = src.profile.copy()
        labels = src.read(1)
        transform = src.transform
        crs = src.crs

    name_to_id = {mc.name: cid for cid, mc in MODEL_LEGEND.items()}
    pts = gpd.read_file(al_path, layer=layer).to_crs(crs)
    pts = pts[pts["model_class"].fillna("").str.strip().isin(name_to_id)]

    out = labels.copy()
    n = 0
    for _, p in pts.iterrows():
        disc = p.geometry.buffer(AL_DISC_M)
        cid = name_to_id[p["model_class"].strip()]
        mask = rasterize([(disc, 1)], out_shape=labels.shape, transform=transform,
                         fill=0, dtype=np.uint8).astype(bool)
        out[mask] = cid
        n += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    changed = int((out != labels).sum())
    print(f"Inyección de active learning -> {out_path}")
    print(f"  puntos aplicados: {n} | píxeles de verdad exacta: {changed:,}")
    return out_path


def flag_truth_chips(chips_csv: Path, point_specs: list[tuple[Path, str]], ortho_path: Path,
                     out_csv: Path) -> Path:
    """Marca en el índice de chips cuáles contienen algún punto de verdad (campo/AL),
    para sobre-muestrearlos en el entrenamiento."""
    import pandas as pd
    from shapely.geometry import box

    chips = pd.read_csv(chips_csv)
    with rasterio.open(ortho_path) as o:
        tr, crs = o.transform, o.crs

    pts = []
    for path, layer in point_specs:
        if path and Path(path).exists():
            g = gpd.read_file(path, layer=layer).to_crs(crs)
            if "model_class" in g.columns:
                g = g[g["model_class"].fillna("").str.strip() != ""]
            pts.append(g[["geometry"]])
    allp = gpd.GeoDataFrame(__import__("pandas").concat(pts, ignore_index=True), crs=crs) if pts else None

    has_truth = np.zeros(len(chips), dtype=int)
    if allp is not None and len(allp):
        sidx = allp.sindex
        for i, row in chips.iterrows():
            c0, r0, sz = int(row["col_off"]), int(row["row_off"]), int(row["size_px"])
            x0, y0 = tr * (c0, r0)
            x1, y1 = tr * (c0 + sz, r0 + sz)
            geom = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            hit = list(sidx.intersection(geom.bounds))
            if hit and allp.iloc[hit].intersects(geom).any():
                has_truth[i] = 1
    chips["has_truth"] = has_truth
    chips.to_csv(out_csv, index=False)
    print(f"chips con verdad puntual: {int(has_truth.sum())}/{len(chips)} -> {out_csv}")
    return out_csv


def uncertainty_sample(n_candidates: int, n_select: int, seed: int, out_path: Path) -> Path:
    """Muestreo por incertidumbre con Clay para la ronda de etiquetado del usuario.

    Genera candidatos aleatorios en el AOI, predice con Clay, mide incertidumbre
    (entropía de la prob media en el parche central) y selecciona los más inciertos,
    estratificados por zona N-S (para cubrir el sur, que rinde peor) y con prioridad a
    la frontera Cultivos/Pastos/Arbustal (donde está el error de destino).
    """
    import sys
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    from rasterio.windows import Window
    from shapely.geometry import Point

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline_v3.train_clay_lora import INPUT_PX, ClaySegmenter
    from pipeline_v3.legend import num_model_classes

    WS = Path(__file__).resolve().parents[1] / "workspace_rural_aoi"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_cls = num_model_classes()
    model = ClaySegmenter(n_cls)
    state = torch.load(WS / "models/clay_lora/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"]); model.to(device).eval()

    aoi = gpd.read_file(WS / "data/aoi/aoi_efectivo.gpkg", layer="aoi_efectivo").to_crs(9377)
    aoi_geom = aoi.union_all()
    minx, miny, maxx, maxy = aoi.total_bounds
    h = maxy - miny
    t1, t2 = maxy - h / 3, maxy - 2 * h / 3
    # exclusión de validación y campo
    excl = []
    for p, lyr in [(WS / "deliverables/val_set_2025.gpkg", "val_set_2025"),
                   (WS / "deliverables/val_set_2025_complemento.gpkg", "val_set_complemento"),
                   (WS / "data/field/puntos_campo_2026.gpkg", "field_usable")]:
        if p.exists():
            try:
                excl.append(gpd.read_file(p, layer=lyr).to_crs(9377))
            except Exception:
                pass
    import pandas as pd
    excl_union = gpd.GeoDataFrame(pd.concat(excl, ignore_index=True), crs=9377).buffer(60).union_all() if excl else None

    rng = np.random.default_rng(seed)
    CHIP = 512
    cand = []
    with rasterio.open(WS / "data/prepared/ortho_2025_aoi_efectivo.tif") as ortho:
        attempts = 0
        while len(cand) < n_candidates and attempts < n_candidates * 30:
            attempts += 1
            x = float(rng.uniform(minx, maxx)); y = float(rng.uniform(miny, maxy))
            pt = Point(x, y)
            if not aoi_geom.contains(pt):
                continue
            if excl_union is not None and excl_union.contains(pt):
                continue
            r, c = ortho.index(x, y)
            r0, c0 = r - CHIP // 2, c - CHIP // 2
            if r0 < 0 or c0 < 0 or r0 + CHIP > ortho.height or c0 + CHIP > ortho.width:
                continue
            win = Window(c0, r0, CHIP, CHIP)
            bgrn = ortho.read([3, 2, 1, 4], window=win).astype(np.float32)
            if not np.any(bgrn):
                continue
            tin = torch.from_numpy(bgrn / 255.0)[None]
            tin = F.interpolate(tin, size=(INPUT_PX, INPUT_PX), mode="bilinear", align_corners=False)
            with torch.no_grad(), torch.amp.autocast(device):
                logits = model(tin.to(device))
            logits = F.interpolate(logits, size=(CHIP, CHIP), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, 1)[0, :, CHIP // 2 - 8:CHIP // 2 + 8, CHIP // 2 - 8:CHIP // 2 + 8]
            pmean = probs.mean(dim=(1, 2)).cpu().numpy()
            ent = float(-(pmean * np.log(pmean + 1e-9)).sum())
            pred = int(pmean.argmax())
            zona = "norte" if y > t1 else ("centro" if y > t2 else "sur")
            cand.append({"x": x, "y": y, "entropy": ent, "pred": pred, "zona": zona})
            if len(cand) % 200 == 0:
                print(f"  candidatos {len(cand)}/{n_candidates}", flush=True)

    df = pd.DataFrame(cand)
    # prioridad: frontera (pred en herbáceas) pesa más en la incertidumbre
    df["score"] = df["entropy"] * np.where(df["pred"].isin(list(OVERRIDABLE)), 1.3, 1.0)
    # selección estratificada por zona (reparto equilibrado para cubrir centro/sur)
    per_zone = max(1, n_select // 3)
    picks = []
    for zona in ("norte", "centro", "sur"):
        sub = df[df["zona"] == zona].sort_values("score", ascending=False)
        picks.append(sub.head(per_zone))
    sel = pd.concat(picks).reset_index(drop=True)

    from pipeline_v3.legend import MODEL_LEGEND as ML
    gdf = gpd.GeoDataFrame(
        {
            "sample_id": [f"AL_{i:04d}" for i in range(len(sel))],
            "zona": sel["zona"].values,
            "pred_modelo": [ML[int(p)].name for p in sel["pred"]],
            "entropy": sel["entropy"].round(3).values,
            "model_class": "",   # <- lo rellena el usuario
            "notas": "",
        },
        geometry=[Point(x, y) for x, y in zip(sel["x"], sel["y"])],
        crs=9377,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, layer="active_round1", driver="GPKG")
    print(f"\nronda de incertidumbre: {len(gdf)} puntos -> {out_path}")
    print(gdf.groupby(["zona", "pred_modelo"]).size().unstack(fill_value=0).to_string())
    print("\nEtiquetar 'model_class' en QGIS. Clases:", ", ".join(mc.name for mc in ML.values()))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Active learning")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inject-field")
    pi.add_argument("--weak", required=True, type=Path)
    pi.add_argument("--field", required=True, type=Path)
    pi.add_argument("--val", type=Path, default=None)
    pi.add_argument("--val-extra", type=Path, default=None)
    pi.add_argument("--out", required=True, type=Path)

    pu = sub.add_parser("uncertainty-sample")
    pu.add_argument("--candidates", type=int, default=2500)
    pu.add_argument("--select", type=int, default=150)
    pu.add_argument("--seed", type=int, default=123)
    pu.add_argument("--out", required=True, type=Path)

    pa = sub.add_parser("inject-active")
    pa.add_argument("--weak", required=True, type=Path)
    pa.add_argument("--al", required=True, type=Path)
    pa.add_argument("--layer", default="active_round1")
    pa.add_argument("--out", required=True, type=Path)

    pf = sub.add_parser("flag-chips")
    pf.add_argument("--chips", required=True, type=Path)
    pf.add_argument("--ortho", required=True, type=Path)
    pf.add_argument("--field", type=Path, default=None)
    pf.add_argument("--al", type=Path, default=None)
    pf.add_argument("--al-layer", default="active_round1")
    pf.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.cmd == "inject-field":
        inject_field(args.weak, args.field, [args.val, args.val_extra], args.out)
    elif args.cmd == "uncertainty-sample":
        uncertainty_sample(args.candidates, args.select, args.seed, args.out)
    elif args.cmd == "inject-active":
        inject_active_points(args.weak, args.al, args.layer, args.out)
    elif args.cmd == "flag-chips":
        specs = []
        if args.field:
            specs.append((args.field, "field_usable"))
        if args.al:
            specs.append((args.al, args.al_layer))
        flag_truth_chips(args.chips, specs, args.ortho, args.out)


if __name__ == "__main__":
    main()
