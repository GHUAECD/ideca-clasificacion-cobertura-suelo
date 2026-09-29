#!/usr/bin/env bash
# Secuencia completa del flujo que produjo la entrega de junio de 2026.
#
# Es una guía ejecutable: cada paso puede correrse por separado. Los pasos 4 y 5
# requieren trabajo manual en QGIS (etiquetar puntos) y no se pueden automatizar.
# Tiempo aproximado en una GPU de 24 GB: entrenamiento ~1 h; inferencia del AOI
# completo varias horas (reanudable por tesela).
#
# Uso:  RUTA_INSUMOS=/ruta/a/insumos PROYECTO_GEE=mi-proyecto bash scripts/ejecutar_flujo_final.sh
set -euo pipefail

: "${RUTA_INSUMOS:?Defina RUTA_INSUMOS (carpeta con orto, MDT, límite, CLC 2016 y puntos de campo)}"
: "${PROYECTO_GEE:?Defina PROYECTO_GEE (proyecto de Google Cloud habilitado para Earth Engine)}"

WS=workspace_rural_aoi
ORTO_FUENTE="$RUTA_INSUMOS/ortofoto_2025_0_5m.tif"
MDT_FUENTE="$RUTA_INSUMOS/mdt_10m.tif"
LIMITE="$RUTA_INSUMOS/AOI_BOGOTA_RURAL.shp"
CLC_2016="$RUTA_INSUMOS/coverage_2016_aoi.gpkg"        # capa coverage_2016 con campo CODIGO_ID
PUNTOS_CAMPO="$RUTA_INSUMOS/puntos_rurales_certeza.shp"

# 1. AOI efectivo = huella de datos de la orto ∩ límite oficial
python -m pipeline_v3.aoi_footprint --ortho "$ORTO_FUENTE" --aoi-oficial "$LIMITE" \
    --out "$WS/data/aoi/aoi_efectivo.gpkg"

# 2. Re-recorte de orto y MDT al AOI efectivo + 128 m
python scripts/recortar_insumos.py --ortho "$ORTO_FUENTE" --mdt "$MDT_FUENTE" \
    --aoi "$WS/data/aoi/aoi_efectivo.gpkg" --out-dir "$WS/data/prepared"

# 3. Etiquetas débiles 2025: Dynamic World (moda anual) + clases estables de CLC 2016
#    Requiere haber ejecutado una vez:  earthengine authenticate
python -m pipeline_v3.weak_labels_gee export --ortho "$WS/data/prepared/ortho_2025_aoi_efectivo.tif" \
    --project "$PROYECTO_GEE" --out "$WS/data/labels/dw2025_label_10m.tif"
python -m pipeline_v3.weak_labels_gee homologate --dw "$WS/data/labels/dw2025_label_10m.tif" \
    --ortho "$WS/data/prepared/ortho_2025_aoi_efectivo.tif" \
    --out "$WS/data/labels/weak_labels_dw2025_model.tif"
python -m pipeline_v3.label_fusion --weak "$WS/data/labels/weak_labels_dw2025_model.tif" \
    --clc-vector "$CLC_2016" --clc-mapping config/class_mapping.csv \
    --out "$WS/data/labels/weak_labels_fused.tif"

# 4. Set de validación 2025 (solo para medir; nunca entra al entrenamiento)
python -m pipeline_v3.validation_set --ortho "$WS/data/prepared/ortho_2025_aoi_efectivo.tif" \
    --prior "$WS/data/labels/weak_labels_dw2025_model.tif" --per-class 60 \
    --clc-vector "$CLC_2016" --clc-mapping config/class_mapping.csv \
    --supplement "Frailejonal:60,Humedal:40" --out "$WS/deliverables/val_set_2025.gpkg"
echo ">> MANUAL: abrir val_set_2025.gpkg sobre la orto en QGIS y diligenciar 'model_class' por punto."
echo ">> MANUAL: en la entrega se añadió un complemento centro/sur (8 puntos por clase y zona)"
echo "           en val_set_2025_complemento.gpkg (capa val_set_complemento)."

# 5. Puntos de campo (verificación de vecindad)
python -m pipeline_v3.field_points --shp "$PUNTOS_CAMPO" --aoi "$WS/data/aoi/aoi_efectivo.gpkg" \
    --out "$WS/data/field/puntos_campo_2026.gpkg"

# 6. Bloques espaciales (80/10/10) e índice de chips de 512 px
python -m pipeline_v3.prepare --ortho "$WS/data/prepared/ortho_2025_aoi_efectivo.tif" \
    --labels "$WS/data/labels/weak_labels_fused.tif" --aoi "$WS/data/aoi/aoi_efectivo.gpkg" \
    --out-dir "$WS/data/intermediate"

# 7. Entrenamiento SegFormer (configuración de la entrega)
python -m pipeline_v3.train_segformer --labels "$WS/data/labels/weak_labels_fused.tif" \
    --hf-model nvidia/mit-b2 --epochs 12 --batch-size 8 --lr 6e-5 --bosque-discount 0.5 --seed 42 \
    --out-dir "$WS/models/segformer_v3"

# 8. Validación contra la verdad 2025
#    ("all" = puntos + vecindad en un solo validation_2025.json; por separado se sobrescriben)
python -m pipeline_v3.validate_2025 all --checkpoint "$WS/models/segformer_v3/best.pt"

# 9. Inferencia sobre todo el AOI (reanudable) y mosaico
python -m pipeline_v3.produce_dl run --checkpoint "$WS/models/segformer_v3/best.pt"
python -m pipeline_v3.produce_dl mosaic --sieve 1000

# 10. Refinamiento dirigido Arbustal -> Pastos y herbáceo, vectorización y homologación
python -m pipeline_v3.refine_arbustal run
python -m pipeline_v3.refine_arbustal finalize --sieve 1000

# 11. Shapefiles de entrega
python scripts/exportar_entrega_shp.py \
    --gpkg "$WS/deliverables/producto_dl_refinado/cobertura_2025_dl_refinado_destinos.gpkg" \
    --out-dir "$WS/deliverables/entrega_shp"
