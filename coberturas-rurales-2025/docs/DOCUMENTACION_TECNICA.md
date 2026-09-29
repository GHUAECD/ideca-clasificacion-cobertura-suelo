# Documentación técnica — pipeline_v3 (coberturas 2025 Sumapaz)

Documentación interna y reproducible del pipeline. No es el documento de entrega
(ese es `workspace_rural_aoi/deliverables/Metodologia_Coberturas_2025_Sumapaz.md`).

> **Nota de publicación (septiembre de 2026).** Las rutas locales y el identificador del proyecto de Earth Engine se reemplazaron por marcadores (`<RUTA_INSUMOS>`, `<PROYECTO_GEE>`). El modelo entregado es `segformer_v3/best.pt` (11-jun-2026, etiquetas fusionadas, sin aprendizaje activo); los resultados de aprendizaje activo de la §8 corresponden a Clay + LoRA. El re-recorte de la §1 y la exportación a shapefile quedaron como scripts en `scripts/`.

---

## 0. Entorno

- **SO/shell:** Windows + WSL2 (Ubuntu). Todo el cómputo corre en WSL.
- **Conda env:** `coberturas-2025` (ver `environment.yml`) (Python 3.11).
  ```bash
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate coberturas-2025
  ```
- **GPU:** NVIDIA RTX 4500 Ada (24 GB), CUDA 12.8.
- **Librerías clave:** torch 2.11.0+cu128, transformers 5.6, terratorch 1.2.8, peft 0.19.1,
  segment-geospatial/samgeo 1.3.2 (SAM 2), earthengine-api 1.7.30, geedim 2.0.0,
  rasterio, GDAL 3.10, geopandas, scikit-learn, scipy 1.17.1, opencv.
- **Proyecto GEE:** un proyecto de Google Cloud habilitado para Earth Engine (`--project <PROYECTO_GEE>`).
- **Raíz del proyecto:** la raíz de este repositorio
  - código: `pipeline_v3/`  · workspace: `workspace_rural_aoi/`

### Convención de ejecución (importante)
Lanzar siempre con env activado y módulo `-m`:
```bash
cd <raiz_proyecto>
python -m pipeline_v3.<modulo> <args>
```

---

## 1. Insumos y AOI

| Archivo | Ruta | Notas |
|---|---|---|
| Orto fuente | `<RUTA_INSUMOS>/ortofoto_2025_0_5m.tif` | ~60 GB, 83.671×179.178, 4 bandas RGBN, 8 bits |
| MDT fuente | `<RUTA_INSUMOS>/mdt_10m.tif` | 10 m |
| Límite oficial | `<RUTA_INSUMOS>/AOI_BOGOTA_RURAL.shp` | 1.073 km² |
| CLC 2016 (AOI) | `workspace_rural_aoi/data/prepared/coverage_2016_aoi.gpkg` (capa `coverage_2016`) | con `CODIGO_ID` |
| class_mapping | `workspace_rural_aoi/config/class_mapping.csv` | CODIGO_ID→macro_id |
| Puntos campo OTC | `<RUTA_INSUMOS>/puntos_rurales_certeza.shp` | 789 pts (552 en AOI, 351 utilizables) |

### AOI efectivo — `aoi_footprint.py`
Footprint de datos válidos (banda>0) de la orto fuente ∩ límite oficial.
- Decima la orto a 5 m (gdal.Translate), vectoriza máscara con `rasterio.features.shapes`,
  filtra polígonos < 100.000 m², simplifica 2 m.
- Resultado: `workspace_rural_aoi/data/aoi/aoi_efectivo.gpkg` (capas `footprint`, `aoi_efectivo`) + `.shp`.
- **Hallazgo:** la orto cubre el 100 % del límite oficial (0 km² sin imagen); la orto *preparada* anterior carecía de 28 km² → re-recorte obligatorio desde la fuente.
```bash
python -m pipeline_v3.aoi_footprint --ortho "<RUTA_INSUMOS>/ortofoto_2025_0_5m.tif" \
  --aoi-oficial "<RUTA_INSUMOS>/AOI_BOGOTA_RURAL.shp" \
  --out workspace_rural_aoi/data/aoi/aoi_efectivo.gpkg
```

### Re-recorte de insumos
Cutline = `aoi_efectivo` + buffer 128 m (`warp_raster` de `pipeline_v1/common.py`, `resample near` orto / `bilinear` MDT, `target_aligned_pixels`).
- `ortho_2025_aoi_efectivo.tif` (83.984×179.491, 0,5 m) · `mdt_2025_aoi_efectivo.tif` (4.200×8.975, 10 m).
- **Gotcha GDAL 3.10:** `WarpOptions(width=None)` se serializa como literal `'None'` y falla → pasar `width=0, height=0`.

---

## 2. Leyenda y homologación — `legend.py`
- `MODEL_LEGEND`: 9 clases (0 Construido, 1 Cultivos, 2 Pastos_y_herbaceo, 3 Bosque,
  4 Arbustal_y_secundaria, 5 Frailejonal, 6 Suelo_desnudo_roca, 7 Humedal, 8 Agua).
- `IGNORE_INDEX = 255`.
- `model_to_destination_id()` → 6 destinos. Banderas:
  - `PARAMO_HERBAZAL_AS_PROTECTED = False` (Herbazal→Agropecuario; voltear a Suelo Protegido si Catastro lo decide).
  - `HUMEDAL_AGUA_DESTINO = "Suelo_Protegido"` (observación técnica del 15-may-2026).
- `MACRO10_TO_MODEL` (CLC-10→v3) y `DYNAMICWORLD_TO_MODEL` (DW 0-8→v3).
- `write_tables()` → CSVs en `pipeline_v3/config/`.
```bash
python -m pipeline_v3.legend   # regenera tablas
```

---

## 3. Etiquetas débiles 2025

### 3.1 Dynamic World — `weak_labels_gee.py`
- Colección `GOOGLE/DYNAMICWORLD/V1`, `filterDate(2025)`, `reduce(mode)`, banda `label`.
- **Offset +1**: la clase DW 0=water choca con el nodata=0 de geedim → se descarga con `.add(1)` (clases 1-9) y la LUT de homologación descuenta el offset.
- **CRS de descarga = EPSG:32618** (EE no parsea EPSG:9377); homologate reproyecta a la grilla de la orto (nearest).
```bash
python -m pipeline_v3.weak_labels_gee export --ortho .../ortho_2025_aoi_efectivo.tif \
  --project <PROYECTO_GEE> --out .../data/labels/dw2025_label_10m.tif
python -m pipeline_v3.weak_labels_gee homologate --dw .../dw2025_label_10m.tif \
  --ortho .../ortho_2025_aoi_efectivo.tif --out .../data/labels/weak_labels_dw2025_model.tif
```

### 3.2 Fusión con CLC estable — `label_fusion.py`
DW **no** da Frailejonal (→0 px) y casi no da Humedal (2.904 px). Se inyectan desde CLC 2016
(Frailejonal=CODIGO 321114→clase 5; Humedal=411/412/413x→7), erosión 10 m, sobre la base DW.
```bash
python -m pipeline_v3.label_fusion --weak .../weak_labels_dw2025_model.tif \
  --clc-vector .../coverage_2016_aoi.gpkg --clc-mapping .../class_mapping.csv \
  --out .../data/labels/weak_labels_fused.tif
```
Distribución resultante (px 10 m): Bosque 28,8M · Arbustal 13,5M · Construido 6,1M ·
Pastos 5,0M · Cultivos 2,58M · Frailejonal 1,41M · Suelo 334k · Agua 288k · Humedal 74k.

---

## 4. Verdad de validación — `validation_set.py`
- Muestreo estratificado por el prior (60/clase) + suplemento CLC (Frailejonal 60, Humedal 40)
  + complemento centro/sur (8/clase/zona). Total ~492 pts; el usuario etiquetó `model_class`.
- Set **sagrado** (solo validación). Verdad efectiva evaluable: ~437 pts dentro del AOI.
- **Gotcha CRS:** comparar CRS por `to_epsg()`/`to_authority()`, no por igualdad de objeto (WKT vs `EPSG:9377`).

---

## 5. Preparación de entrenamiento — `prepare.py`
- Bloques 1.024 m con split espacial 80/10/10 (por bloque, evita fuga).
- Chips 512 px, stride 512; `MIN_AOI_FRACTION=0.5`, `MIN_LABEL_FRACTION=0.3`.
- Etiquetas alineadas al vuelo desde 10 m (`read_label_window_aligned`, nearest) — sin intermedio de 15 Gpx.
- Salida: `v3_blocks.gpkg`, `v3_chips.csv` (16.136 chips; 12.980 train).

---

## 6. Objetos — `objects_sam.py`
- SAM 2 (`sam2-hiera-large`) vía samgeo. `points_per_side=24` (producto) / 48 (prueba),
  `min_mask_region_area=400`. ~0,8 s/tesela @24, 2048 px.
- `masks_to_segment_ids` (grandes primero, pequeñas pisan), `regularize_window` (clase mayoritaria por segmento, vectorizado con bincount).
- **Advertencia ignorable:** `cannot import name '_C' from 'sam2'` (post-proc no compilado; no afecta resultados).

---

## 7. Modelos

### 7.1 SegFormer — `train_segformer.py`
- Backbone `nvidia/mit-b2` preentrenado; `adapt_input_channels` a **8 canales**
  (R,G,B,NIR,NDVI,NDWI,elev,pendiente; `feature_stack` de `pipeline_v2/utils.py`).
- `FEATURES`: elevation_min 2400, max 4200, slope_scale 1.0.
- Chips 512 px, batch 8, lr 6e-5 cosine + warmup, 12 épocas, pérdida `CE(class_weights, ignore=255) + 0.5·Dice`,
  `WeightedRandomSampler`, `--bosque-discount 0.5` (DW infla Bosque en páramo).
- Mejor val macro-F1 (vs etiqueta débil) ≈ 0,502 (época 8). Checkpoint `models/segformer_v3/best.pt` (`{"model": state_dict}`).
```bash
python -m pipeline_v3.train_segformer --labels .../weak_labels_fused.tif --epochs 12 \
  --out-dir .../models/segformer_v3
```

### 7.2 Clay + LoRA — `train_clay_lora.py`
- `timm_clay_v1_base` (terratorch), `bands=[BLUE,GREEN,RED,NIR_NARROW]`, **256 px** (ViT con posición fija; falla a 512),
  LoRA r=16 α=32 + cabeza conv → **3,1 % de params entrenables**. batch 16, 12 épocas. Mejor ≈ 0,507.

### 7.3 OBIA + Random Forest — `obia_rf.py`
- Features por segmento: NDVI(media/std), textura std brillo, **GLCM** (contrast/homog/energy), NIR, NDWI, brillo, elev, prior.
- GLCM contrast separa Pastos (0,32) de Arbustal (0,70). RF (300-400 árboles, class_weight balanced).
- Producto RF tiene mayor accuracy puntual (0,753) pero **hereda el grano 10 m de DW** (feature `prior` dominante) → descartado por fidelidad temática.

---

## 8. Active learning — `active_learning.py`
- `uncertainty-sample`: entropía de la prob media en parche; 150 pts estratificados por zona → `active_round1.gpkg`. Usuario etiquetó 136 (corrigió al modelo 65 %).
- `inject-field`: anillos 12-40 m alrededor de puntos OTC (saltan la vía), solo sobre clases `{1,2,4}`, excluye <30 m de validación.
- `inject-active`: discos de 15 m (verdad exacta).
- `flag-chips` + `--boost-truth` en el trainer: sobre-muestreo de chips con verdad.
- **Resultado:** AL con anillos de campo (ruidoso, boost ×8) regresó a 0,682; AL limpio (solo discos, boost ×3) recuperó a 0,714. No superó el techo.

---

## 9. Producto e inferencia

### 9.1 Producto DL fino (ELEGIDO) — `produce_dl.py`
- Inferencia SegFormer por teselas: tile 2048, **halo 128**, chip 512, stride 256, **mezcla cosine** (`cosine_weight`) → sin costuras. Reanudable por tesela (manifest.json).
- `mosaic`: BuildVRT → recorte AOI → **sieve ligero** (`--sieve 1000` px ≈ 0,025 ha) → vectorización + homologación.
```bash
python -m pipeline_v3.produce_dl run --out-dir .../deliverables/producto_dl
python -m pipeline_v3.produce_dl mosaic --out-dir .../deliverables/producto_dl --sieve 1000
```

### 9.2 Producto grueso (referencia) — `produce_map.py`
Prior fusionado regularizado con SAM (`--classifier prior|rf`). `MAX_SEG_FRAC=0.20`
(segmentos gigantes conservan el prior píxel a píxel → evita flips temáticos por tesela).
`prior_window_aligned` usa `rasterio.band` (grilla global) para evitar astillas entre teselas.

### 9.3 Refinamiento dirigido — `refine_arbustal.py`
- Regla única: `cover==4 (Arbustal) AND std_brillo_local<9 AND ndvi_local<0.22 → 2 (Pastos)`.
  Ventana 32 px (~16 m), `scipy.ndimage.uniform_filter`. Orto reproyectada a la grilla del cover por bloque (alineación exacta).
- Conservador (calibrado con verdad: ~43 % captura herbazal / ~11 % FP). **Única transición 4→2 (11.064 ha)**, verificado.
```bash
python -m pipeline_v3.refine_arbustal run        # genera ráster refinado
python -m pipeline_v3.refine_arbustal finalize   # sieve + vectoriza + homologa
```
- **Producto final:** `deliverables/producto_dl_refinado/` (.tif + .gpkg capas cobertura/destinos).

---

## 10. Validación — `validate_2025.py`
- `points`: vs set fotointerpretado (parche modal 5×5), métricas 9 clases + 6 destinos + por zona N-S.
- `vicinity`: chequeo de vecindad con puntos OTC (radio 75 m).

### Tablero comparativo (concordancia destinos vs ~437 pts)
| Enfoque | Destinos |
|---|---|
| DW crudo 10 m | 0,686 |
| DW+CLC fusionado (prior) | 0,735 |
| SAM 2 + prior | 0,709 |
| SegFormer (val pts / producto AOI) | 0,705 / 0,712 |
| Clay v1.5 + LoRA | 0,716 |
| Clay + AL (anillos campo, ruido) | 0,682 |
| Clay + AL limpio | 0,714 |
| OBIA + RF (GLCM) | 0,753 |
| **DL fino + refinamiento (final)** | **≈0,71** (Agropecuario recall 0,41→0,48) |

**Conclusión:** techo ~0,71-0,735 impuesto por la supervisión débil, no por el modelo. El accuracy puntual favorece a los productos gruesos (heredan grano DW); la selección del producto fue por **fidelidad temática** verificada en terreno.

---

## 11. Gotchas operativos (lecciones)

- **WSL se reinicia periódicamente** (idle): logs/outputs **siempre en el disco de Windows montado (`/mnt/<unidad>`)**, nunca `/tmp` (se borra). Jobs largos **reanudables por tesela**.
- **Variables `$VAR` se vacían** en `bash -lc` multilínea vía `wsl.exe` → usar **scripts `.sh` en disco** y `bash script.sh`.
- **`pgrep` se auto-detecta** → usar patrón `[x]xxx` (`pgrep -f '[p]roduce_dl'`).
- **Monitores con grep de tildes** ('área') no hacen match (codificación) → usar ASCII (`rea por destino`).
- **EE no parsea EPSG:9377** → descargar en 32618 y reproyectar local.
- **DW water=0 vs geedim nodata=0** → offset +1 en la descarga.
- **rasterio `tiled=True` sin blockxsize** → bloque = ancho completo (no múltiplo de 16, error) → fijar 256.
- **Reproyección por ventana con `round_offsets()`** → astillas/franjas entre teselas → usar `rasterio.band` (grilla global).
- **Regularización SAM con segmentos gigantes** (páramo homogéneo) → flips temáticos por tesela → cap `MAX_SEG_FRAC=0.20`.
- **Nombre de driver `'ESRI Shapefile'`** (con espacio) se rompe vía wsl.exe → script `.sh` + comillas simples.

---

## 12. Estructura de archivos generados

```
workspace_rural_aoi/
  data/
    aoi/aoi_efectivo.gpkg, ortho_fuente_decimada_5m.tif
    prepared/ortho_2025_aoi_efectivo.tif, mdt_2025_aoi_efectivo.tif
    labels/weak_labels_dw2025_model.tif, weak_labels_fused.tif, weak_labels_al.tif
    field/puntos_campo_2026.gpkg (field_usable)
    intermediate/v3_blocks.gpkg, v3_chips*.csv, rf_train_glcm.csv
  models/segformer_v3/, clay_lora*/, obia_rf.joblib
  deliverables/
    producto_dl/            (DL fino)
    producto_dl_refinado/   (FINAL: cobertura_2025_dl_refinado_aoi.tif + _destinos.gpkg)
    entrega_shp/            (shapefiles de entrega)
    producto_priorSAM/, producto_rf/  (referencia/comparación)
    validation_2025*.json
```
