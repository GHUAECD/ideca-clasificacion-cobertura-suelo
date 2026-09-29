# Coberturas de la tierra 2025 – zona rural de Bogotá (Sumapaz)

Código del flujo que produjo la **capa de coberturas de la tierra 2025** del área rural de Sumapaz
(9 clases) y su **homologación a 6 destinos económicos catastrales**, entregada en junio de 2026
como insumo de trabajo para la homologación de destinos económicos.

- **Entidad:** Unidad Administrativa Especial de Catastro Distrital (UAECD), con definiciones de coberturas de IDECA.
- **Versión:** 3.0.0 (entrega de junio de 2026). Ver [CHANGELOG.md](CHANGELOG.md).
- **Naturaleza del producto:** insumo de trabajo; **no** constituye cartografía oficial.
- **Este repositorio contiene solo código y tablas de configuración.** No incluye imágenes, capas,
  puntos de campo ni modelos entrenados (ver [Datos](#datos)).

## Resultado de la versión entregada

| Indicador | Valor |
|---|---|
| Concordancia en destinos económicos (437 puntos verificados sobre la orto 2025) | **70,5 %** SegFormer; **≈71 %** tras el refinamiento |
| Concordancia a nivel de 9 coberturas | 60,0 % |
| Concordancia por zona (destinos) | norte 70,6 % · centro 72,7 % · sur 66,2 % |
| Área clasificada | 107.298 ha (límite rural oficial, EPSG:9377) |

Área por destino: Forestales 78.365 ha · Agropecuario 21.967 ha · Agrícola 4.326 ha ·
No aplica 1.400 ha · Suelo protegido (63) 1.106 ha · Tierras improductivas 134 ha.

La validación usa puntos **independientes del entrenamiento**, fotointerpretados sobre la orto 2025.
El detalle de métodos, comparación de enfoques y limitaciones está en [docs/METODOLOGIA.md](docs/METODOLOGIA.md).

## Cómo funciona (flujo final)

```
Orto 2025 (0,5 m, RGB+NIR) + MDT (10 m) + límite rural oficial
  │ 1. aoi_footprint          AOI efectivo = huella de la orto ∩ límite oficial
  │ 2. recortar_insumos       re-recorte de orto y MDT al AOI + 128 m
  │ 3. weak_labels_gee        Dynamic World 2025 (moda anual) → leyenda de 9 clases
  │    label_fusion           + frailejonal y humedal desde CLC 2016 (coberturas estables)
  │ 4. validation_set         puntos de validación estratificados (se etiquetan en QGIS)
  │ 5. field_points           puntos de campo → verificación de vecindad
  │ 6. prepare                bloques espaciales 80/10/10 + chips de 512 px
  │ 7. train_segformer        SegFormer (MiT-B2), 8 canales: RGB, NIR, NDVI, NDWI, elevación, pendiente
  │ 8. validate_2025          concordancia contra los puntos 2025 (9 clases y 6 destinos)
  │ 9. produce_dl             inferencia por teselas con halo y mezcla coseno → mosaico, recorte, sieve
  │10. refine_arbustal        regla única Arbustal → Pastos y herbáceo (textura y NDVI bajos)
  │11. exportar_entrega_shp   Coberturas_2025_Sumapaz.shp y Destinos_Economicos_2025_Sumapaz.shp
```

La secuencia exacta de comandos está en [scripts/ejecutar_flujo_final.sh](scripts/ejecutar_flujo_final.sh).
Los parámetros de la ejecución entregada son los valores por defecto del código, salvo `--labels`
(etiquetas fusionadas) en el entrenamiento:

| Paso | Parámetros |
|---|---|
| Entrenamiento | `nvidia/mit-b2`, 12 épocas, lote 8, lr 6e-5 (coseno con calentamiento), CE ponderada + 0,5·Dice, descuento Bosque 0,5, semilla 42 |
| Inferencia | teselas de 2.048 px, halo 128 px, chips de 512 px con paso 256 y mezcla coseno |
| Posproceso | sieve de 1.000 px (≈0,025 ha) |
| Refinamiento | ventana 32 px (~16 m); desviación estándar del brillo < 9 y NDVI < 0,22 |

### Módulos experimentales

Se conservan porque sustentan la comparación de enfoques reportada en la metodología, pero **no
intervienen en la capa entregada**: `objects_sam.py` y `produce_map.py` (SAM 2 + guía),
`obia_rf.py` (OBIA + Random Forest), `train_clay_lora.py` y `eval_clay_points.py` (Clay v1.5 + LoRA)
y `active_learning.py` (aprendizaje activo, evaluado con Clay).

`pipeline_v1/` y `pipeline_v2/` contienen solo los módulos de etapas anteriores que el flujo final
reutiliza (malla de bloques, recorte, pila de variables, pérdida Dice y adaptación de canales).

## Instalación

Requiere Linux (se ejecutó en Ubuntu sobre WSL2) y, para entrenar e inferir en tiempos razonables,
una GPU NVIDIA (se usó una de 24 GB con CUDA 12.8).

```bash
conda env create -f environment.yml
conda activate coberturas-2025
python -m pytest -q          # pruebas rápidas, no requieren datos
```

Para descargar Dynamic World se necesita una cuenta de Google Earth Engine y un proyecto de
Google Cloud habilitado: `earthengine authenticate` y luego `--project <PROYECTO_GEE>`.

## Datos

Los datos **no** se publican en este repositorio. El código espera esta estructura, relativa a la raíz:

```
workspace_rural_aoi/
  data/aoi/aoi_efectivo.gpkg                 (capa aoi_efectivo)
  data/prepared/ortho_2025_aoi_efectivo.tif  (RGB+NIR, 0,5 m, uint8)
  data/prepared/mdt_2025_aoi_efectivo.tif    (10 m)
  data/labels/weak_labels_fused.tif          (etiquetas débiles, 10 m)
  data/field/puntos_campo_2026.gpkg          (capa field_usable)
  data/intermediate/v3_blocks.gpkg, v3_chips.csv
  deliverables/val_set_2025.gpkg, val_set_2025_complemento.gpkg
  models/segformer_v3/best.pt
```

| Insumo | Fuente |
|---|---|
| Ortofoto 2025 (0,5 m, 4 bandas) y MDT (10 m) | UAECD / IDECA |
| Límite rural oficial (`AOI_BOGOTA_RURAL.shp`) | UAECD |
| Interpretación Corine Land Cover 2016 (campo `CODIGO_ID`) | IDECA; homologación en [config/class_mapping.csv](config/class_mapping.csv) |
| Dynamic World V1 (`GOOGLE/DYNAMICWORLD/V1`) | Google y World Resources Institute, licencia CC BY 4.0 |
| Puntos de campo con registro fotográfico | Observatorio Técnico Catastral (OTC), UAECD |
| Puntos de validación 2025 | Fotointerpretación del proyecto |

Las solicitudes de acceso a los datos o al modelo entrenado se tramitan con la UAECD.

## Tablas de configuración

- [pipeline_v3/config/model_legend.csv](pipeline_v3/config/model_legend.csv): 9 clases del modelo, color y destino.
- [pipeline_v3/config/destinations.csv](pipeline_v3/config/destinations.csv): destinos económicos y código catastral conocido.
- [pipeline_v3/config/dynamicworld_to_model.csv](pipeline_v3/config/dynamicworld_to_model.csv) y [macro10_to_model.csv](pipeline_v3/config/macro10_to_model.csv): homologaciones de entrada.
- [pipeline_v3/config/email_homologacion.csv](pipeline_v3/config/email_homologacion.csv): tabla original de homologación, conservada por trazabilidad.

Las dos decisiones de criterio temático están como banderas al inicio de
[pipeline_v3/legend.py](pipeline_v3/legend.py): `PARAMO_HERBAZAL_AS_PROTECTED` (herbazal de páramo como
Suelo protegido; `False` en la entrega) y `HUMEDAL_AGUA_DESTINO` (`Suelo_Protegido`). Si se modifican,
regenere las tablas con `python -m pipeline_v3.legend`.

## Limitaciones y problemas conocidos

1. **Validación no probabilística.** Los puntos se distribuyeron por clase y zona. Sirven para medir
   concordancia, pero no para estimar áreas con intervalos de confianza.
2. **Calibración del refinamiento.** Los umbrales de `refine_arbustal.py` se calibraron con puntos de
   verdad y el script de calibración no se conservó. Si esos puntos incluyen el set de validación, la
   mejora medida después del refinamiento puede estar sobrestimada.
3. **Polígonos diminutos.** El sieve usa conectividad 8 y la vectorización conectividad 4. Por eso
   quedan 9.013 polígonos (17 %) menores de 0,025 ha, que suman 6,4 ha. Corrección sugerida:
   `rasterio.features.shapes(..., connectivity=8)`.
4. **Sin unidad mínima cartografiable.** Solo se aplica el sieve de 0,025 ha: el 86 % de los polígonos
   mide menos de 0,5 ha (3.941 ha en total).
5. **Bordes de tesela.** En algunos sectores se observan límites rectos que coinciden con la malla de
   inferencia de 2.048 px.
6. **Aprendizaje activo no incorporado.** El modelo entregado no usa los 136 puntos revisados; esa
   ronda se evaluó con Clay + LoRA.
7. **Pasos manuales.** El complemento centro/sur del set de validación y el etiquetado de puntos en QGIS
   se hicieron a mano. El re-recorte de insumos y la exportación a shapefile se hicieron de forma
   interactiva; en esta versión quedaron como scripts y se verificaron contra la entrega (misma grilla
   del MDT; shapefile de coberturas idéntico byte a byte).
8. **Índice de chips.** El `v3_chips.csv` usado el 11-jun-2026 se regeneró el 12-jun para los
   experimentos con Clay. `prepare.py` es determinista (semilla 42) y lo reconstruye.
9. **Rutas fijas.** `produce_dl.py`, `refine_arbustal.py`, `obia_rf.py` y `produce_map.py` leen los
   insumos de rutas fijas dentro de `workspace_rural_aoi/`; respete la estructura de [Datos](#datos).

## Estructura del repositorio

```
pipeline_v3/        flujo final y módulos experimentales + config/ (tablas de homologación)
pipeline_v1/        common.py (malla de bloques y recorte) reutilizado por v3
pipeline_v2/        utils.py, segformer_v2.py, config.py reutilizados por v3
config/             class_mapping.csv (códigos CLC 2016 → 10 macroclases)
scripts/            recortar_insumos.py, exportar_entrega_shp.py, ejecutar_flujo_final.sh
docs/               METODOLOGIA.md (documento de entrega) y DOCUMENTACION_TECNICA.md
tests/              pruebas rápidas sin datos
```

## Licencia

**Pendiente de definición por la entidad.** Hasta que se agregue un archivo `LICENSE`, el código no
otorga permisos de reutilización. Los pesos preentrenados de terceros (por ejemplo `nvidia/mit-b2`) y
los modelos derivados de ellos tienen licencias propias que deben revisarse antes de publicarlos.

## Cómo citar

Unidad Administrativa Especial de Catastro Distrital – UAECD (2026). *Coberturas de la tierra 2025 –
zona rural de Bogotá (Sumapaz)*, versión 3.0.0 [código fuente]. Desarrollo técnico: Fabián Guzmán.

## Créditos

- Homologación de coberturas a destinos económicos: Gerencia de Información Catastral, UAECD, con base en las definiciones de coberturas de IDECA.
- Registro de campo: Observatorio Técnico Catastral (OTC), UAECD.
- Productos abiertos: Google Dynamic World (Sentinel-2) y la interpretación Corine Land Cover 2016.
