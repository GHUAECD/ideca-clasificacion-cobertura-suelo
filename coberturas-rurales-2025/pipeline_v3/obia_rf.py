"""Refinamiento OBIA + Random Forest: la clase del segmento sale de features finos.

Reemplaza el "voto del prior DW (10 m)" por un Random Forest entrenado con la verdad
puntual del usuario sobre features de la orto 0,5 m + MDT, que SÍ separan herbazal
(liso, NDVI bajo) de arbustal (rugoso, NDVI alto) y frailejonal (alta elevación) —
justo lo que DW fundía. Reutiliza SAM 2 para los objetos; solo cambia quién decide
la clase de cada objeto.

Features por muestra/segmento: NDVI (media/std), textura (std del brillo), NIR, NDWI,
elevación, pendiente, brillo. Más, como contexto, la clase del prior fusionado.

Este módulo: (1) extrae features en los puntos de verdad, (2) entrena/valida el RF con
validación cruzada, (3) expone predict_segments(...) para el runner del producto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_v3.legend import MODEL_LEGEND, IGNORE_INDEX, destination_of  # noqa: E402

WS = ROOT / "workspace_rural_aoi"
ORTHO = WS / "data/prepared/ortho_2025_aoi_efectivo.tif"
MDT = WS / "data/prepared/mdt_2025_aoi_efectivo.tif"
PRIOR = WS / "data/labels/weak_labels_fused.tif"
FEAT_NAMES = ["ndvi", "ndvi_std", "textura", "nir", "ndwi", "brillo",
              "glcm_contrast", "glcm_homog", "glcm_energy", "elev", "prior"]
WIN_R = 16  # radio ventana de muestreo (~8 m), suficiente para GLCM


def _glcm_feats(brillo: np.ndarray) -> tuple[float, float, float]:
    """Contraste, homogeneidad y energía GLCM del brillo (estructura: leñoso vs herbáceo)."""
    from skimage.feature import graycomatrix, graycoprops
    q = np.clip(brillo / 16, 0, 15).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 2], levels=16,
                        symmetric=True, normed=True)
    return (float(graycoprops(glcm, "contrast").mean()),
            float(graycoprops(glcm, "homogeneity").mean()),
            float(graycoprops(glcm, "energy").mean()))


def features_from_arrays(ortho4: np.ndarray, elev: float, prior_val: int) -> list[float]:
    """Vector de features a partir de un parche RGBN (4,h,w), elevación y clase prior."""
    r, g, b, nir = ortho4.astype(np.float32)
    ndvi = (nir - r) / np.clip(nir + r, 1, None)
    ndwi = (g - nir) / np.clip(g + nir, 1, None)
    brillo = ortho4[:3].mean(0)
    gc, gh, ge = _glcm_feats(brillo)
    return [float(ndvi.mean()), float(ndvi.std()), float(brillo.std()),
            float(nir.mean()), float(ndwi.mean()), float(brillo.mean()),
            gc, gh, ge, float(elev), float(prior_val)]


def _truth_points(exact_only: bool = True) -> "gpd.GeoDataFrame":
    import geopandas as gpd
    specs = [(WS / "deliverables/val_set_2025.gpkg", "val_set_2025"),
             (WS / "deliverables/val_set_2025_complemento.gpkg", "val_set_complemento"),
             (WS / "deliverables/active_round1.gpkg", "active_round1")]
    parts = []
    for path, layer in specs:
        try:
            g = gpd.read_file(path, layer=layer)
        except Exception:
            g = gpd.read_file(path)
        g = g[g["model_class"].fillna("").str.strip() != ""][["model_class", "geometry"]]
        g["src"] = layer
        parts.append(g)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))


def build_training_table() -> pd.DataFrame:
    import geopandas as gpd
    name_to_id = {mc.name: cid for cid, mc in MODEL_LEGEND.items()}
    pts = _truth_points()
    rows = []
    with rasterio.open(ORTHO) as o, rasterio.open(MDT) as m, rasterio.open(PRIOR) as pr:
        pts = pts.to_crs(o.crs)
        for _, p in pts.iterrows():
            rr, cc = o.index(p.geometry.x, p.geometry.y)
            if not (WIN_R <= rr < o.height - WIN_R and WIN_R <= cc < o.width - WIN_R):
                continue
            arr = o.read([1, 2, 3, 4], window=Window(cc - WIN_R, rr - WIN_R, 2 * WIN_R, 2 * WIN_R))
            if not np.any(arr):
                continue
            mr, mc = m.index(p.geometry.x, p.geometry.y)
            try:
                elev = float(m.read(1, window=Window(mc, mr, 1, 1))[0, 0])
            except Exception:
                elev = 3200.0
            prr, prc = pr.index(p.geometry.x, p.geometry.y)
            try:
                pv = int(pr.read(1, window=Window(prc, prr, 1, 1))[0, 0])
            except Exception:
                pv = IGNORE_INDEX
            feats = features_from_arrays(arr, elev, pv)
            rows.append({**dict(zip(FEAT_NAMES, feats)),
                         "y": name_to_id[p["model_class"].strip()], "src": p["src"],
                         "x": p.geometry.x, "yy": p.geometry.y})
    return pd.DataFrame(rows)


def evaluate_cv(df: pd.DataFrame, n_splits: int = 5) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    X = df[FEAT_NAMES].to_numpy()
    y = df["y"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cm = np.zeros((len(MODEL_LEGEND), len(MODEL_LEGEND)), dtype=np.int64)
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        pred = rf.predict(X[te])
        for t, p in zip(y[te], pred):
            cm[t, p] += 1

    acc9 = np.trace(cm) / cm.sum()
    # destino
    dests = sorted({mc.destination for mc in MODEL_LEGEND.values()})
    di = {d: i for i, d in enumerate(dests)}
    cmd = np.zeros((len(dests), len(dests)), dtype=np.int64)
    for t in range(len(MODEL_LEGEND)):
        for p in range(len(MODEL_LEGEND)):
            cmd[di[destination_of(t)], di[destination_of(p)]] += cm[t, p]
    accd = np.trace(cmd) / cmd.sum()
    print(f"RF {n_splits}-fold CV  | n={int(cm.sum())} | acc 9 clases {acc9:.3f} | acc destinos {accd:.3f}")
    print("recall por clase del modelo:")
    for i, mc in MODEL_LEGEND.items():
        s = cm[i].sum()
        if s:
            print(f"  {mc.name:24s} {cm[i, i] / s:.2f}  (n={int(s)})")
    # importancia de features (entrena en todo)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=-1).fit(X, y)
    imp = sorted(zip(FEAT_NAMES, rf.feature_importances_), key=lambda kv: -kv[1])
    print("importancia features:", {k: round(v, 2) for k, v in imp})


def train_full(df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(df[FEAT_NAMES].to_numpy(), df["y"].to_numpy())
    return rf


def save_model(out_path: Path = WS / "models/obia_rf.joblib") -> Path:
    """Entrena el RF con TODA la verdad y lo guarda para el runner del producto."""
    import joblib
    df = build_training_table()
    rf = train_full(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"rf": rf, "feat_names": FEAT_NAMES}, out_path)
    print(f"RF entrenado con {len(df)} muestras -> {out_path}")
    return out_path


def classify_segments(seg_ids: np.ndarray, ortho4: np.ndarray, elev_grid: np.ndarray,
                      prior_grid: np.ndarray, rf, ignore: int = IGNORE_INDEX) -> np.ndarray:
    """Clase por segmento vía RF sobre features agregados de cada segmento.

    seg_ids: (H,W) ids (0 = sin segmento). ortho4: (4,H,W). elev_grid/prior_grid: (H,W).
    Devuelve mapa (H,W) de clases. Píxeles sin segmento -> prior.
    """
    from skimage.feature import graycomatrix, graycoprops

    out = prior_grid.copy()
    r, g, b, nir = ortho4.astype(np.float32)
    ndvi = (nir - r) / np.clip(nir + r, 1, None)
    ndwi = (g - nir) / np.clip(g + nir, 1, None)
    brillo = ortho4[:3].mean(0)

    ids = np.unique(seg_ids)
    ids = ids[ids > 0]
    if ids.size == 0:
        return out
    feats, valid_ids = [], []
    for sid in ids:
        mask = seg_ids == sid
        if mask.sum() < 16:
            continue
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub_b = brillo[y0:y1, x0:x1]
        q = np.clip(sub_b / 16, 0, 15).astype(np.uint8)
        q[~mask[y0:y1, x0:x1]] = 0
        try:
            glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 2], levels=16,
                                symmetric=True, normed=True)
            gc = float(graycoprops(glcm, "contrast").mean())
            gh = float(graycoprops(glcm, "homogeneity").mean())
            ge = float(graycoprops(glcm, "energy").mean())
        except Exception:
            gc, gh, ge = 0.0, 1.0, 1.0
        pv = prior_grid[mask]
        pv = int(np.bincount(pv[pv != ignore]).argmax()) if (pv != ignore).any() else ignore
        feats.append([float(ndvi[mask].mean()), float(ndvi[mask].std()), float(brillo[mask].std()),
                      float(nir[mask].mean()), float(ndwi[mask].mean()), float(brillo[mask].mean()),
                      gc, gh, ge, float(elev_grid[mask].mean()), float(pv)])
        valid_ids.append(sid)
    if not feats:
        return out
    preds = rf.predict(np.array(feats))
    lut = {int(s): int(p) for s, p in zip(valid_ids, preds)}
    for sid, cls in lut.items():
        out[seg_ids == sid] = cls
    return out


if __name__ == "__main__":
    df = build_training_table()
    print(f"muestras de verdad con features: {len(df)}")
    print(df["y"].map({c: m.name for c, m in MODEL_LEGEND.items()}).value_counts().to_string())
    print()
    evaluate_cv(df)
