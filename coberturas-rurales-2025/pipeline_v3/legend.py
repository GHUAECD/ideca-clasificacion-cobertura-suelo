"""Leyenda orientada a destinos económicos y tablas de homologación (Fase 0).

Diseño "hacia atrás" desde los 6 destinos económicos catastrales. Toda la lógica de
homologación se ancla en la **tabla de homologación remitida por Catastro Bogotá** (la propuesta
experta de la Gerencia de Información Catastral) más la **observación técnica del 15-may-2026**
para Sumapaz (Humedal y Agua → Suelo Protegido 63, no Espacio Público 66).

Las decisiones que son de criterio experto y no de modelo se dejan como banderas
configurables al inicio del archivo, para que el usuario/Catastro las confirme sin
tocar el resto del código. No se inventa ningún criterio de homologación aquí.

Tablas que produce (CSV versionados en ``pipeline_v3/config/``):
  - ``model_legend.csv``          clases que predice el modelo (id, nombre, paleta, destino)
  - ``destinations.csv``          destinos económicos y su código catastral (donde se conoce)
  - ``macro10_to_model.csv``      remapeo de las 10 macroclases CLC-2016 → clase del modelo
  - ``dynamicworld_to_model.csv`` Dynamic World (9 clases) → clase del modelo (etiqueta débil)
  - ``email_homologacion.csv``    tabla original del correo (10 clases → destino) para trazabilidad
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Banderas de criterio experto (confirmar con Catastro; NO son del modelo)
# ---------------------------------------------------------------------------

# El correo homologa "Herbazal" → Agropecuario de forma genérica. Pero en Sumapaz
# la mayor parte del herbazal es páramo (p. ej. CLC 321111 "Herbazal denso de tierra
# firme", >24.000 ha), que ecológicamente corresponde a área protegida, no a uso
# agropecuario. Si esta bandera es True, el herbazal por encima de PARAMO_ELEV_M se
# trata como vegetación de páramo (destino Suelo Protegido). Si es False, se respeta
# literalmente el correo (todo Herbazal → Agropecuario).
PARAMO_HERBAZAL_AS_PROTECTED = False  # default: respetar el correo; flag para el experto
PARAMO_ELEV_M = 3300.0  # umbral altitudinal de páramo (sólo se usa si la bandera es True)

# Observación técnica (15-may-2026): en Sumapaz, Humedal y Agua → Suelo Protegido (63).
HUMEDAL_AGUA_DESTINO = "Suelo_Protegido"  # alternativa histórica del correo: "Espacio_Publico"


# ---------------------------------------------------------------------------
# Destinos económicos catastrales
# ---------------------------------------------------------------------------
# Códigos catastrales: sólo se fijan los que el correo menciona explícitamente
# (Espacio Público = 66, Suelo Protegido = 63). El resto queda en None a la espera
# de confirmación de Catastro — no se inventan códigos.

@dataclass(frozen=True)
class Destination:
    name: str
    catastral_code: int | None
    note: str


DESTINATIONS: dict[str, Destination] = {
    "No_Aplica": Destination("No_Aplica", None, "Territorios artificializados; el predio no recibe destino de cobertura."),
    "Agricola": Destination("Agricola", None, "Cultivos transitorios y permanentes."),
    "Agropecuario": Destination("Agropecuario", None, "Pastos, mosaicos y herbáceo manejado."),
    "Forestales": Destination("Forestales", None, "Vegetación leñosa: bosque, arbustal/secundaria, frailejonal."),
    "Tierras_Improductivas": Destination("Tierras_Improductivas", None, "Suelo desnudo, roca, afloramientos."),
    "Suelo_Protegido": Destination("Suelo_Protegido", 63, "Protección ambiental (corrección Sumapaz): humedales y cuerpos de agua."),
    "Espacio_Publico": Destination("Espacio_Publico", 66, "Destino original del correo para humedal/agua, sustituido por Suelo Protegido en Sumapaz."),
}


# ---------------------------------------------------------------------------
# Leyenda del modelo: clases que el SegFormer/Clay predice
# ---------------------------------------------------------------------------
# Se eligen para ser (a) separables en RGB+NIR 0.5 m (+ elevación/pendiente del MDT) y
# (b) homologables a un único destino. Frailejonal se mantiene como clase propia porque,
# aunque espectralmente se parece al herbazal de páramo, el MDT (elevación) y su textura
# moteada permiten distinguirlo, y cambia de destino (Forestales vs Agropecuario).
#
# Nota: Dynamic World NO tiene clase frailejonal; la etiqueta débil de páramo cae en
# "Herbazal". Frailejonal se aprende del set de active learning + prior altitudinal/CLC.

@dataclass(frozen=True)
class ModelClass:
    id: int
    name: str
    palette: str  # hex para visualización
    destination: str


def _humedal_agua_dest() -> str:
    return HUMEDAL_AGUA_DESTINO


MODEL_LEGEND: dict[int, ModelClass] = {
    0: ModelClass(0, "Construido", "#c8001a", "No_Aplica"),
    1: ModelClass(1, "Cultivos", "#ffd37f", "Agricola"),
    2: ModelClass(2, "Pastos_y_herbaceo", "#a3cc51", "Agropecuario"),
    3: ModelClass(3, "Bosque", "#1b5e20", "Forestales"),
    4: ModelClass(4, "Arbustal_y_secundaria", "#4c9a2a", "Forestales"),
    5: ModelClass(5, "Frailejonal", "#9b59b6", "Forestales"),
    6: ModelClass(6, "Suelo_desnudo_roca", "#9c9c9c", "Tierras_Improductivas"),
    7: ModelClass(7, "Humedal", "#3aa6b9", _humedal_agua_dest()),
    8: ModelClass(8, "Agua", "#1f6fb2", _humedal_agua_dest()),
}

IGNORE_INDEX = 255


# ---------------------------------------------------------------------------
# Remapeo: 10 macroclases CLC-2016 (v1/v2) → clase del modelo v3
# ---------------------------------------------------------------------------
# Permite reusar etiquetas/predicciones existentes en la nueva leyenda. Macroclase ->
# (id macro original en common.MACRO_CLASSES) según pipeline_v1.
#   0 Artificial 1 Cultivos 2 Pastos_y_mosaicos 3 Bosque 4 Arbustal_y_secundaria
#   5 Herbazal 6 Frailejonal 7 Humedal 8 Suelo_desnudo_rocoso 9 Agua
MACRO10_TO_MODEL: dict[int, int] = {
    0: 0,  # Artificial            -> Construido
    1: 1,  # Cultivos              -> Cultivos
    2: 2,  # Pastos_y_mosaicos     -> Pastos_y_herbaceo
    3: 3,  # Bosque                -> Bosque
    4: 4,  # Arbustal_y_secundaria -> Arbustal_y_secundaria
    5: 2,  # Herbazal              -> Pastos_y_herbaceo (Agropecuario, salvo override páramo)
    6: 5,  # Frailejonal           -> Frailejonal
    7: 7,  # Humedal               -> Humedal
    8: 6,  # Suelo_desnudo_rocoso  -> Suelo_desnudo_roca
    9: 8,  # Agua                  -> Agua
}


# ---------------------------------------------------------------------------
# Dynamic World (etiqueta débil 2025) → clase del modelo v3
# ---------------------------------------------------------------------------
# Clases Dynamic World V1 (banda "label", 0-8):
#   0 water 1 trees 2 grass 3 flooded_vegetation 4 crops
#   5 shrub_and_scrub 6 built 7 bare 8 snow_and_ice
DYNAMICWORLD_NAMES: dict[int, str] = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}

DYNAMICWORLD_TO_MODEL: dict[int, int] = {
    0: 8,  # water              -> Agua
    1: 3,  # trees              -> Bosque
    2: 2,  # grass              -> Pastos_y_herbaceo  (páramo cae aquí; frailejonal no lo da DW)
    3: 7,  # flooded_vegetation -> Humedal
    4: 1,  # crops              -> Cultivos
    5: 4,  # shrub_and_scrub    -> Arbustal_y_secundaria
    6: 0,  # built              -> Construido
    7: 6,  # bare               -> Suelo_desnudo_roca
    8: 6,  # snow_and_ice       -> Suelo_desnudo_roca (nieve permanente inexistente en Sumapaz; raro)
}


# ---------------------------------------------------------------------------
# Tabla original del correo (10 clases macro → destino) — para trazabilidad
# ---------------------------------------------------------------------------
EMAIL_HOMOLOGACION: dict[str, str] = {
    "Artificial": "No_Aplica",
    "Cultivos": "Agricola",
    "Pastos_y_mosaicos": "Agropecuario",
    "Bosque": "Forestales",
    "Arbustal_y_secundaria": "Forestales",
    "Herbazal": "Agropecuario",
    "Frailejonal": "Forestales",
    "Humedal": "Espacio_Publico",  # corregido a Suelo_Protegido en Sumapaz
    "Suelo_desnudo_rocoso": "Tierras_Improductivas",
    "Agua": "Espacio_Publico",     # corregido a Suelo_Protegido en Sumapaz
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def destination_of(model_class_id: int) -> str:
    """Destino económico de una clase del modelo."""
    return MODEL_LEGEND[model_class_id].destination


def model_to_destination_id() -> dict[int, str]:
    return {cid: mc.destination for cid, mc in MODEL_LEGEND.items()}


def num_model_classes() -> int:
    return len(MODEL_LEGEND)


def write_tables(out_dir: Path) -> dict[str, Path]:
    """Escribe todas las tablas de homologación como CSV versionados."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    legend_df = pd.DataFrame(
        [(mc.id, mc.name, mc.palette, mc.destination) for mc in MODEL_LEGEND.values()],
        columns=["model_id", "model_class", "palette", "destination"],
    )
    paths["model_legend"] = out_dir / "model_legend.csv"
    legend_df.to_csv(paths["model_legend"], index=False, encoding="utf-8")

    dest_df = pd.DataFrame(
        [(d.name, d.catastral_code, d.note) for d in DESTINATIONS.values()],
        columns=["destination", "catastral_code", "note"],
    )
    paths["destinations"] = out_dir / "destinations.csv"
    dest_df.to_csv(paths["destinations"], index=False, encoding="utf-8")

    macro_df = pd.DataFrame(
        [(k, v, MODEL_LEGEND[v].name) for k, v in sorted(MACRO10_TO_MODEL.items())],
        columns=["macro10_id", "model_id", "model_class"],
    )
    paths["macro10_to_model"] = out_dir / "macro10_to_model.csv"
    macro_df.to_csv(paths["macro10_to_model"], index=False, encoding="utf-8")

    dw_df = pd.DataFrame(
        [
            (k, DYNAMICWORLD_NAMES[k], v, MODEL_LEGEND[v].name, MODEL_LEGEND[v].destination)
            for k, v in sorted(DYNAMICWORLD_TO_MODEL.items())
        ],
        columns=["dw_id", "dw_class", "model_id", "model_class", "destination"],
    )
    paths["dynamicworld_to_model"] = out_dir / "dynamicworld_to_model.csv"
    dw_df.to_csv(paths["dynamicworld_to_model"], index=False, encoding="utf-8")

    email_df = pd.DataFrame(
        [(k, v) for k, v in EMAIL_HOMOLOGACION.items()],
        columns=["macro_class", "destino_correo"],
    )
    paths["email_homologacion"] = out_dir / "email_homologacion.csv"
    email_df.to_csv(paths["email_homologacion"], index=False, encoding="utf-8")

    return paths


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    written = write_tables(here / "config")
    print(f"Leyenda del modelo: {num_model_classes()} clases -> {len(set(model_to_destination_id().values()))} destinos")
    for name, path in written.items():
        print(f"  {name:24s} -> {path}")
