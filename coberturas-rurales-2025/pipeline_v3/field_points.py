"""Integración de los puntos de campo fotográficos (recorrido rural en sitio).

Fuente: ``puntos_rurales_certeza.shp`` — 789 puntos GPS de precisión (hrms < 1 m) con
foto en sitio y campo ``cobertura`` en **texto libre** diligenciado durante el recorrido
(p. ej. "papa", "Raygraz", "frailejones"), mezclado con notas de logística del
recorrido ("vía en mal estado", "perros", "fin via").

Dos características definen el rol de estos datos en el proceso:

1. **El punto está sobre la vía, no sobre la cobertura**: las coordenadas son del lugar
   desde donde se tomó la foto (la carretera), y la cobertura descrita es lo que se ve
   *alrededor*. Por eso NO se usan como verdad puntual píxel-a-píxel (eso lo da el set
   de validación del usuario). Sus roles:
     a. **Chequeo de vecindad** (validación complementaria): la clase declarada debe
        existir en el mapa predicho dentro de un radio R del punto (~75 m).
     b. **Semillas de active learning**: segmentos SAM cercanos compatibles con la
        clase declarada.
     c. **Apoyo fotográfico**: resolver puntos dudosos del set de validación cercanos.

2. **El texto libre exige normalización**: este módulo lo mapea a la leyenda del modelo
   con reglas por palabra clave (cultivos por especie, pastos por variedad, etc.) y
   descarta explícitamente la logística. Lo no reconocido queda sin clase (usable solo
   por su foto).

Salida: GeoPackage con dos capas:
  - ``field_usable``: puntos dentro del AOI real con clase normalizada.
  - ``field_all``: todos los puntos con su estado (para trazabilidad/QC).
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .legend import MODEL_LEGEND, destination_of

# ---------------------------------------------------------------------------
# Normalización del texto libre -> clase del modelo
# ---------------------------------------------------------------------------
# Reglas por palabra clave sobre texto sin tildes y en minúsculas. El orden importa:
# la primera regla que aplica gana. Diseñadas a partir de los valores reales del
# shapefile (campaña abril-2026, Sumapaz).

_KEYWORD_RULES: list[tuple[tuple[str, ...], str]] = [
    # — Cultivos (especies y estados de siembra vistos en campo, con typos reales)
    (("papa", "arveja", "arvej", "arrvj", "alpiste", "maiz", "fresa", "haba", "cebolla",
      "zanahoria", "fríjol", "frijol", "navo", "nabo", "cultivo", "cultuvo", "cultiv",
      "siembra", "sembrad", "cosecha", "preparacion", "para siembra", "invernadero",
      "barbecho", "arado", "arada", "alistamiento"), "Cultivos"),
    # — Pastos y herbáceo manejado (variedades forrajeras, con typos reales)
    (("pasto", "raygra", "raigra", "reygra", "rygra", "faygra", "poa", "kikuyo", "grama",
      "potrero", "herbazal", "pajonal", "diente de leon"), "Pastos_y_herbaceo"),
    # — Frailejonal (antes que arbustal: "frailejones con arbustos" es frailejonal)
    (("frailejon", "frailejón"), "Frailejonal"),
    # — Bosque y plantaciones
    (("pino", "eucalipto", "acacia", "cipres", "ciprés", "bosque", "plantacion",
      "arbolado", "arboles", "árboles", "forestal"), "Bosque"),
    # — Arbustal / vegetación secundaria (retamo = arbusto invasor; chusque)
    (("retamo", "arbustal", "matorral", "chusque", "chuscal", "rastrojo"), "Arbustal_y_secundaria"),
    # — Agua y humedal
    (("cuerpo de agua", "laguna", "quebrada", "rio", "río", "embalse", "agua"), "Agua"),
    (("humedal", "turbera", "pantano", "juncal"), "Humedal"),
    # — Suelo desnudo / roca
    (("roca", "suelo desnudo", "erosion", "erosión", "cantera", "arenal", "carcava",
      "deslizamiento", "tierra desnuda"), "Suelo_desnudo_roca"),
]

# Términos que son logística/observaciones del recorrido, NUNCA cobertura.
_LOGISTICS_TERMS = (
    "via", "vía", "fin via", "cerrado", "cerrada", "porton", "portón", "puente",
    "perros", "aviso", "recorrido", "final", "privado", "inviable", "desconocido",
    "no paso", "zona militar", "sin camino", "dificil", "difícil", "deslizamos",
    "espinoso", "fin de mapa", "barro", "casa", "bodega", "porteria", "portería",
    "carretera", "camino", "acceso", "lluvias", "mal estado", "terios", "cerca",
    "cerramiento", "maleza", "monte", "vegetacion", "vegetación", "p", "poq", "detalle",
)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_cover_text(raw: str) -> tuple[str | None, str]:
    """Devuelve (clase_modelo | None, motivo). Reglas por palabra clave."""
    if raw is None:
        return None, "vacio"
    text = _strip_accents(str(raw).strip().lower())
    if not text:
        return None, "vacio"

    for keywords, model_class in _KEYWORD_RULES:
        for kw in keywords:
            if _strip_accents(kw) in text:
                return model_class, f"kw:{kw}"

    for term in _LOGISTICS_TERMS:
        if _strip_accents(term) == text or _strip_accents(term) in text:
            return None, f"logistica:{term}"

    return None, "no_reconocido"


def process_field_points(
    shp_path: Path,
    aoi_path: Path,
    out_path: Path,
    target_crs: int = 9377,
) -> gpd.GeoDataFrame:
    pts = gpd.read_file(shp_path).to_crs(target_crs)
    aoi = gpd.read_file(aoi_path).to_crs(target_crs)
    aoi_geom = aoi.union_all()

    results = [normalize_cover_text(c) for c in pts["cobertura"]]
    pts["model_class"] = [r[0] or "" for r in results]
    pts["norm_rule"] = [r[1] for r in results]
    pts["model_id"] = pts["model_class"].map(
        {mc.name: cid for cid, mc in MODEL_LEGEND.items()}
    ).fillna(-1).astype(int)
    pts["destination"] = pts["model_id"].map(
        lambda i: destination_of(int(i)) if int(i) >= 0 else ""
    )
    pts["inside_aoi"] = pts.within(aoi_geom)
    pts["usable"] = (pts["model_id"] >= 0) & pts["inside_aoi"]

    keep_cols = [
        "uid", "prof", "recorrido", "cobertura", "model_class", "model_id",
        "destination", "norm_rule", "inside_aoi", "usable", "fecha", "hora",
        "hrms", "foto", "foto_path", "geometry",
    ]
    pts = pts[[c for c in keep_cols if c in pts.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts.to_file(out_path, layer="field_all", driver="GPKG")
    pts[pts["usable"]].to_file(out_path, layer="field_usable", driver="GPKG")

    print(f"Puntos de campo: {len(pts)} totales -> {out_path}")
    print(f"  dentro AOI real : {int(pts['inside_aoi'].sum())}")
    print(f"  con clase usable: {int(pts['usable'].sum())}  (capa field_usable)")
    usable = pts[pts["usable"]]
    if len(usable):
        print("Distribución de clases normalizadas (usables):")
        print(usable["model_class"].value_counts().to_string())
    no_rec = pts[(pts["norm_rule"] == "no_reconocido")]["cobertura"].value_counts()
    if len(no_rec):
        print(f"\nTextos no reconocidos ({no_rec.sum()} pts) — revisar si alguno es cobertura:")
        print(no_rec.head(20).to_string())
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalización de puntos de campo fotográficos")
    ap.add_argument("--shp", required=True, type=Path)
    ap.add_argument("--aoi", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    process_field_points(args.shp, args.aoi, args.out)


if __name__ == "__main__":
    main()
