"""Generador del set de validación 2025 estratificado para auto-etiquetado (Fase 0).

Produce una muestra de **puntos** estratificada por clase sobre la ortofoto 2025, como
GeoPackage editable. El usuario abre la orto + esta capa en QGIS y rellena el campo
``model_class`` con la cobertura real 2025 que observa en cada punto (sólo clases fáciles
de reconocer sin ser experto). Ese set es la **verdad de validación**: nunca entra a
entrenamiento, y las métricas se reportan a nivel de destino.

La estratificación usa un ráster *prior* de clases (idealmente Dynamic World 2025 ya
homologado a la leyenda del modelo). El prior sólo sirve para repartir los puntos entre
clases y como referencia de control de calidad — NO es la etiqueta final; ésa la pone el
usuario. Si no hay prior, se muestrea de forma espacialmente uniforme dentro del AOI.

Eficiente: lee el prior (10 m, pequeño) en memoria; sólo toca la orto (25 Gpx) en
ventanas de 1 px para descartar puntos sobre nodata.

Uso:
    python -m pipeline_v3.validation_set \
        --ortho workspace_rural_aoi/data/prepared/ortho_2025_aoi.tif \
        --prior workspace_rural_aoi/data/labels/weak_labels_dw2025_model.tif \
        --per-class 60 --out workspace_rural_aoi/deliverables/val_set_2025.gpkg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point

from .legend import MODEL_LEGEND, IGNORE_INDEX, destination_of


def _crs_provably_different(a, b) -> bool:
    """True sólo si se puede probar que dos CRS difieren.

    Evita el falso negativo de comparar un CRS en forma WKT contra el mismo CRS en
    forma ``EPSG:xxxx`` (caso típico en rasterio). Si no se puede normalizar ninguno,
    no se afirma diferencia (el prior real se reproyecta a la grilla de la orto en
    ``weak_labels_gee.py``).
    """
    if a == b:
        return False
    try:
        ea, eb = a.to_epsg(), b.to_epsg()
        if ea is not None and eb is not None:
            return ea != eb
    except Exception:
        pass
    try:
        aa, ab = a.to_authority(), b.to_authority()
        if aa and ab:
            return aa != ab
    except Exception:
        pass
    return False


def _valid_in_ortho(ortho: rasterio.io.DatasetReader, x: float, y: float) -> bool:
    """True si el punto (x, y) cae sobre píxel de datos (no nodata) en la orto."""
    try:
        row, col = ortho.index(x, y)
    except Exception:
        return False
    if not (0 <= row < ortho.height and 0 <= col < ortho.width):
        return False
    win = rasterio.windows.Window(col, row, 1, 1)
    arr = ortho.read(window=win)  # (bands, 1, 1)
    if arr.size == 0:
        return False
    nodata = ortho.nodata
    if nodata is not None and np.all(arr == nodata):
        return False
    # orto uint8 RGB+NIR: descartar negros puros (relleno de recorte)
    return bool(np.any(arr != 0))


def sample_stratified(
    ortho_path: Path,
    prior_path: Path | None,
    per_class: int,
    seed: int = 42,
    max_attempts_factor: int = 40,
) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    with rasterio.open(ortho_path) as ortho:
        ortho_crs = ortho.crs

        if prior_path is not None:
            with rasterio.open(prior_path) as prior:
                prior_arr = prior.read(1)
                prior_transform = prior.transform
                prior_crs = prior.crs
            if _crs_provably_different(prior_crs, ortho_crs):
                raise ValueError(
                    f"El prior (EPSG:{prior_crs.to_epsg()}) y la orto (EPSG:{ortho_crs.to_epsg()}) "
                    "tienen CRS distintos. Reproyecta el prior antes (ver weak_labels_gee.py)."
                )
            class_ids = [cid for cid in MODEL_LEGEND if cid != IGNORE_INDEX]
            for cid in class_ids:
                rows, cols = np.where(prior_arr == cid)
                if rows.size == 0:
                    continue
                take = min(per_class, rows.size)
                budget = take * max_attempts_factor
                idx_pool = rng.permutation(rows.size)[:budget]
                got = 0
                for k in idx_pool:
                    if got >= take:
                        break
                    # centro del píxel prior + jitter dentro del píxel
                    px_x, px_y = xy(prior_transform, int(rows[k]), int(cols[k]))
                    res_x = abs(prior_transform.a)
                    res_y = abs(prior_transform.e)
                    jx = px_x + float(rng.uniform(-0.4, 0.4)) * res_x
                    jy = px_y + float(rng.uniform(-0.4, 0.4)) * res_y
                    if _valid_in_ortho(ortho, jx, jy):
                        records.append(_make_record(cid, jx, jy, prior_known=True))
                        got += 1
        else:
            # muestreo espacial uniforme dentro de los límites de la orto
            left, bottom, right, top = ortho.bounds
            target = per_class * len([c for c in MODEL_LEGEND])
            attempts = 0
            while len(records) < target and attempts < target * max_attempts_factor:
                attempts += 1
                x = float(rng.uniform(left, right))
                y = float(rng.uniform(bottom, top))
                if _valid_in_ortho(ortho, x, y):
                    records.append(_make_record(None, x, y, prior_known=False))

    gdf = gpd.GeoDataFrame(records, crs=ortho_crs)
    gdf = gdf.reset_index(drop=True)
    gdf["sample_id"] = [f"VAL_{i:05d}" for i in range(len(gdf))]
    cols = ["sample_id", "prior_id", "prior_class", "prior_source", "model_id", "model_class", "destination", "notas", "geometry"]
    return gdf[cols]


def _make_record(prior_cid: int | None, x: float, y: float, prior_known: bool, source: str = "dw2025") -> dict:
    prior_name = MODEL_LEGEND[prior_cid].name if (prior_known and prior_cid is not None) else ""
    return {
        "prior_id": int(prior_cid) if (prior_known and prior_cid is not None) else -1,
        "prior_class": prior_name,   # referencia/QC; NO es la etiqueta final
        "prior_source": source,      # dw2025 | clc2016 | uniforme
        "model_id": -1,              # <- lo rellena el usuario (o vacío)
        "model_class": "",           # <- lo rellena el usuario en QGIS
        "destination": "",           # se deriva luego de model_class
        "notas": "",
        "geometry": Point(x, y),
    }


def supplement_from_clc(
    ortho_path: Path,
    clc_vector: Path,
    clc_mapping_csv: Path,
    wanted: dict[str, int],
    seed: int = 42,
    layer: str = "coverage_2016",
) -> list[dict]:
    """Puntos extra para clases que el prior DW no cubre (p. ej. Frailejonal, Humedal).

    Ubica los puntos dentro de polígonos CLC 2016 de esas clases (rasgos estables a
    9 años: páramo, turberas, lagunas); la clase real 2025 la confirma el usuario.
    """
    name_to_id = {mc.name: cid for cid, mc in MODEL_LEGEND.items()}
    # CLC CODIGO_ID -> macro_id (10 clases v1) -> clase del modelo v3
    from .legend import MACRO10_TO_MODEL

    mapping = pd.read_csv(clc_mapping_csv)[["CODIGO_ID", "macro_id"]].drop_duplicates()
    mapping["model_id"] = mapping["macro_id"].map(MACRO10_TO_MODEL)

    gdf = gpd.read_file(clc_vector, layer=layer)
    gdf = gdf.merge(mapping, on="CODIGO_ID", how="left")

    rng = np.random.default_rng(seed)
    records: list[dict] = []
    with rasterio.open(ortho_path) as ortho:
        if _crs_provably_different(gdf.crs, ortho.crs):
            gdf = gdf.to_crs(ortho.crs)
        for class_name, n_points in wanted.items():
            cid = name_to_id[class_name]
            polys = gdf[gdf["model_id"] == cid]
            if polys.empty:
                print(f"  [aviso] sin polígonos CLC 2016 para {class_name}; se omite")
                continue
            areas = polys.geometry.area.to_numpy()
            weights = areas / areas.sum()
            got, attempts = 0, 0
            while got < n_points and attempts < n_points * 50:
                attempts += 1
                poly = polys.iloc[int(rng.choice(len(polys), p=weights))].geometry
                minx, miny, maxx, maxy = poly.bounds
                x = float(rng.uniform(minx, maxx))
                y = float(rng.uniform(miny, maxy))
                if not poly.contains(Point(x, y)):
                    continue
                if _valid_in_ortho(ortho, x, y):
                    records.append(_make_record(cid, x, y, prior_known=True, source="clc2016"))
                    got += 1
            print(f"  suplemento CLC2016 {class_name}: {got}/{n_points} puntos")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Set de validación 2025 estratificado para auto-etiquetado")
    ap.add_argument("--ortho", required=True, type=Path)
    ap.add_argument("--prior", type=Path, default=None,
                    help="Ráster de clases del modelo para estratificar (p.ej. Dynamic World homologado). Opcional.")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clc-vector", type=Path, default=None,
                    help="GPKG CLC 2016 para suplementar clases ausentes del prior")
    ap.add_argument("--clc-mapping", type=Path, default=None,
                    help="class_mapping.csv (CODIGO_ID -> macro_id)")
    ap.add_argument("--supplement", default="",
                    help="Clases extra desde CLC, p.ej. 'Frailejonal:60,Humedal:30'")
    args = ap.parse_args()

    gdf = sample_stratified(args.ortho, args.prior, args.per_class, args.seed)

    if args.supplement:
        if not (args.clc_vector and args.clc_mapping):
            raise SystemExit("--supplement requiere --clc-vector y --clc-mapping")
        wanted = {}
        for item in args.supplement.split(","):
            name, n = item.split(":")
            wanted[name.strip()] = int(n)
        extra = supplement_from_clc(args.ortho, args.clc_vector, args.clc_mapping, wanted, args.seed)
        if extra:
            extra_gdf = gpd.GeoDataFrame(extra, crs=gdf.crs)
            gdf = gpd.GeoDataFrame(pd.concat([gdf, extra_gdf], ignore_index=True), crs=gdf.crs)
            gdf["sample_id"] = [f"VAL_{i:05d}" for i in range(len(gdf))]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.out, layer="val_set_2025", driver="GPKG")

    print(f"Set de validación: {len(gdf)} puntos -> {args.out}")
    if "prior_class" in gdf and (gdf["prior_class"] != "").any():
        print("Distribución por clase prior (para QC, NO es la etiqueta final):")
        print(gdf["prior_class"].value_counts().to_string())
    print("\nSiguiente paso: abrir la orto + la capa en QGIS y rellenar 'model_class' por punto.")
    print("Clases válidas:", ", ".join(mc.name for mc in MODEL_LEGEND.values()))


if __name__ == "__main__":
    main()
