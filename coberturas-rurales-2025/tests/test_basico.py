"""Pruebas rápidas que no requieren datos, GPU ni conexión.

Ejecutar desde la raíz del repositorio:  python -m pytest -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v1.common import block_grid
from pipeline_v2.utils import cosine_weight, feature_stack
from pipeline_v3 import legend
from pipeline_v3.field_points import normalize_cover_text
from pipeline_v3.objects_sam import masks_to_segment_ids, regularize_window

CONFIG = Path(__file__).resolve().parents[1] / "pipeline_v3" / "config"


def test_leyenda_nueve_clases_seis_destinos():
    assert legend.num_model_classes() == 9
    destinos = set(legend.model_to_destination_id().values())
    assert destinos == {"No_Aplica", "Agricola", "Agropecuario", "Forestales",
                        "Tierras_Improductivas", "Suelo_Protegido"}
    assert legend.destination_of(7) == legend.destination_of(8) == "Suelo_Protegido"
    assert legend.PARAMO_HERBAZAL_AS_PROTECTED is False


def test_tablas_versionadas_coinciden_con_el_codigo(tmp_path):
    escritas = legend.write_tables(tmp_path)
    for ruta in escritas.values():
        nuevo = pd.read_csv(ruta)
        versionado = pd.read_csv(CONFIG / ruta.name)
        pd.testing.assert_frame_equal(nuevo, versionado, check_dtype=False)


def test_normalizacion_texto_de_campo():
    assert normalize_cover_text("papa")[0] == "Cultivos"
    assert normalize_cover_text("Raygraz")[0] == "Pastos_y_herbaceo"
    assert normalize_cover_text("frailejones con arbustos")[0] == "Frailejonal"
    assert normalize_cover_text("vía en mal estado") == (None, "logistica:via")
    assert normalize_cover_text("") == (None, "vacio")


def test_regularizacion_por_objeto():
    pred = np.array([[1, 1, 2],
                     [1, 2, 2],
                     [3, 3, 255]], dtype=np.uint8)
    seg = np.array([[1, 1, 1],
                    [1, 1, 2],
                    [0, 0, 2]], dtype=np.uint32)
    out = regularize_window(pred, seg)
    # segmento 1 -> mayoría clase 1; segmento 2 -> clase 2; sin segmento conserva la clase
    assert out.tolist() == [[1, 1, 1], [1, 1, 2], [3, 3, 255]]


def test_mascaras_sam_pequenas_pisan_a_grandes():
    grande = {"segmentation": np.ones((4, 4), bool), "area": 16}
    chica = np.zeros((4, 4), bool); chica[0, 0] = True
    ids = masks_to_segment_ids([{"segmentation": chica, "area": 1}, grande], (4, 4))
    assert ids[0, 0] == 2 and ids[3, 3] == 1


def test_pila_de_variables_ocho_canales():
    rng = np.random.default_rng(0)
    orto = rng.integers(0, 256, size=(4, 32, 32), dtype=np.uint8)
    elev = rng.uniform(2600, 3800, size=(32, 32)).astype(np.float32)
    x = feature_stack(orto, elev, pixel_size=0.5, elevation_min=2400.0,
                      elevation_max=4200.0, slope_scale=1.0)
    assert x.shape == (8, 32, 32) and x.dtype == np.float32
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_peso_coseno_y_malla_de_bloques():
    w = cosine_weight(512)
    assert w.shape == (512, 512) and np.isclose(w.min(), 0.10)
    grid = block_grid((0.0, 0.0, 2048.0, 1024.0), 1024.0, "EPSG:9377")
    assert len(grid) == 2
