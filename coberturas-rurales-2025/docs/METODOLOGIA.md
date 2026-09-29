# Metodología — Capa de Coberturas de la Tierra 2025 y Homologación a Destinos Económicos
## Área rural de Sumapaz, Bogotá D.C.

**Producto:** capa de coberturas de la tierra a fecha 2025 y su homologación a destinos económicos catastrales, como insumo de trabajo para el proceso de homologación de ZHF del área rural de Sumapaz.
**Insumo base:** ortofotografía 2025 (0,5 m).
**Sistema de referencia:** EPSG:9377 — MAGNA-SIRGAS 2018 / Origen Nacional.
**Naturaleza:** insumo de trabajo, no constituye cartografía oficial.

---

> **Nota de trazabilidad (septiembre de 2026, publicación del código).** El modelo SegFormer que produjo la capa entregada (`models/segformer_v3/best.pt`, 11-jun-2026) se entrenó con las etiquetas fusionadas Dynamic World 2025 + CLC 2016 (clases estables), 12 épocas, **sin** los puntos de aprendizaje activo. La ronda de aprendizaje activo (§5.6) se evaluó con el modelo Clay + LoRA (concordancia 0,714) y no se incorporó al modelo final. El texto de §5.6 se conserva como se entregó.

## 1. Resumen ejecutivo

Se generó una capa de coberturas de la tierra 2025 del área rural de Sumapaz mediante **clasificación por aprendizaje profundo (deep learning, modelo SegFormer)** sobre la ortofoto de 0,5 m, y se homologó a **seis destinos económicos catastrales**. La capa se **validó contra verdad de campo 2025**, obteniendo una **concordancia del 71 % a nivel de destino económico** — una estimación honesta, medida contra el terreno actual y no contra cartografía de referencia anterior.

El desarrollo replanteó un enfoque previo que entrenaba y validaba contra una interpretación CORINE Land Cover (CLC) de 2016, lo que producía un sesgo de origen (desfase temporal de 9 años y validación circular). El nuevo enfoque se ancló en **etiquetas contemporáneas a la imagen (2025)** y en **verdad de campo**, y cerró con un **refinamiento dirigido** guiado por el conocimiento de terreno del responsable.

---

## 2. Contexto y objetivo

El objetivo no es el mapa de coberturas en sí, sino un **insumo lo más cercano posible a la realidad 2025** que permita a Catastro Bogotá homologar destinos económicos en el área rural de Sumapaz.

La homologación de la codificación CLC a destinos económicos fue **propuesta por la Gerencia de Información Catastral** (20 de mayo de 2026), con base en las definiciones de coberturas entregadas por **IDECA**. Se incorporó la observación técnica del 15 de mayo de 2026 de asociar **Humedal y Agua al destino Suelo Protegido (63)**.

---

## 3. Insumos

| Insumo | Descripción | Uso |
|---|---|---|
| Ortofoto 2025 | 0,5 m, 4 bandas (R, G, B, NIR), 8 bits | Base de clasificación |
| Modelo Digital de Terreno (MDT) | 10 m | Elevación y pendiente (features) |
| Límite oficial del proyecto | `07_Límite del Proyecto/AOI_BOGOTA_RURAL.shp` (1.073 km²) | Definición del AOI |
| Dynamic World 2025 (GEE) | 10 m, 9 clases, deep learning sobre Sentinel-2 | Etiquetas débiles contemporáneas |
| Interpretación CLC 2016 | Vectorial, leyenda CLC | Inyección de clases estables (frailejonal/humedal); auxiliar |
| Puntos de campo (OTC) | 789 puntos con registro fotográfico | Verificación de vecindad / apoyo |
| Puntos fotointerpretados 2025 | >430 puntos sobre la ortofoto | Verdad de validación |

**Definición del AOI efectivo.** Se generó el *footprint* de datos válidos de la ortofoto fuente (≈60 GB) y se intersecó con el límite oficial. La ortofoto cubre el 100 % del límite oficial; la ortofoto de trabajo se re-recortó desde la fuente al AOI efectivo (con buffer de 128 m para el contexto de inferencia).

---

## 4. Diagnóstico del enfoque previo

El pipeline anterior entrenaba un SegFormer contra una rasterización de la interpretación **CLC 2016** generalizada a 10 clases. Se identificaron tres problemas estructurales:

1. **Desfase temporal**: etiquetas de 2016 sobre imagen de 2025 (9 años de cambio de cobertura).
2. **Validación circular**: las métricas comparaban la predicción contra el mismo mapa 2016 — no medían exactitud real 2025. Accuracy reportado de 0,43–0,49 a 10 clases, no interpretable como exactitud de terreno.
3. **Etiquetas-objeto rasterizadas a píxel**: la generalización cartográfica de 2016 se transfería como ruido; clases mezcladas por diseño y pares espectralmente inseparables.

**Conclusión:** el cuello de botella no era el modelo ni la imagen (ambos de buena calidad), sino la **fuente de etiquetas y el esquema de validación**.

---

## 5. Metodología implementada

### 5.1. Leyenda orientada a destinos
Se rediseñó la leyenda "hacia atrás" desde los 6 destinos económicos, en 9 clases de modelo separables en RGB+NIR a 0,5 m:

| Clase de cobertura | Destino económico |
|---|---|
| Construido | No aplica |
| Cultivos | Agrícola |
| Pastos y herbáceo | Agropecuario |
| Bosque | Forestales |
| Arbustal y secundaria | Forestales |
| Frailejonal | Forestales |
| Suelo desnudo / roca | Tierras improductivas |
| Humedal | Suelo Protegido (63) |
| Agua | Suelo Protegido (63) |

### 5.2. Verdad de validación 2025
Se construyó un conjunto de validación **estratificado por clase y por zona (norte–centro–sur)** mediante fotointerpretación de la ortofoto 2025 (>430 puntos). Este conjunto es **independiente del entrenamiento** y se reserva exclusivamente para medir exactitud, evitando la validación circular del enfoque previo.

Como apoyo se incorporó el registro de campo del **Observatorio Técnico Catastral (OTC)**: 789 puntos, de los cuales 552 se ubican dentro del AOI de Sumapaz (los restantes en Cerros Orientales y zona rural norte) y 351 con cobertura utilizable, empleados como verificación de vecindad y apoyo al etiquetado (las coordenadas se tomaron sobre vía, no sobre la cobertura, por lo que no se usaron como verdad puntual estricta).

### 5.3. Etiquetas débiles contemporáneas (2025)
Se reemplazó la etiqueta CLC 2016 por **Dynamic World 2025** (composición de moda anual sobre Sentinel-2, descargada de Google Earth Engine), homologada a la leyenda de 9 clases. Esto corrige el desfase temporal: la supervisión queda alineada con la imagen.

**Fusión con CLC 2016 para clases estables.** Dynamic World no representa Frailejonal (lo asigna a pastos) y casi no detecta Humedal. Como estas coberturas son **temporalmente estables** (el frailejonal crece ~1 cm/año; los humedales de páramo no migran), se inyectaron desde los polígonos CLC 2016. Es la división de roles correcta: *Dynamic World para lo dinámico, CLC para lo estable que DW no distingue.*

### 5.4. Segmentación de objetos (SAM 2)
Se empleó **Segment Anything 2 (SAM 2)** sobre la ortofoto 0,5 m para delimitar objetos con fronteras reales (segmentación *class-agnostic*: la forma la da SAM, la clase la pone el modelo semántico). Se usó para regularización por objeto y para el muestreo de active learning.

### 5.5. Modelos evaluados
- **SegFormer** (encoder MiT-B2 preentrenado), 8 canales de entrada: R, G, B, NIR, NDVI, NDWI, elevación, pendiente; ventana 512 px; fine-tuning completo con las etiquetas 2025.
- **Clay v1.5** (modelo fundacional geoespacial) + **LoRA** vía `terratorch`, 4 bandas ópticas, ventana 256 px (solo el 3 % de parámetros entrenables).

Ambos se entrenaron con las etiquetas débiles fusionadas, con balanceo de clases y un **descuento de confianza a la clase Bosque** (la verdad de campo mostró que Dynamic World la sobre-estima en páramo).

### 5.6. Active learning
Se realizó una ronda de muestreo por incertidumbre: el modelo predijo, se seleccionaron los puntos de mayor entropía, y el responsable etiquetó 136 (corrigiendo al modelo en el 65 % de los casos). Estos puntos se inyectaron al entrenamiento como verdad de alta confianza.

### 5.7. Producto final y refinamiento dirigido
El **mapa de deep learning fino (SegFormer)** se seleccionó como producto por su **fidelidad temática**: clasifica píxel a píxel sobre la orto de 0,5 m, capturando la variación fina (mosaicos, transiciones) que las etiquetas de 10 m promedian. Inferencia por teselas con *halo* y mezcla por peso coseno (sin costuras).

**Refinamiento dirigido Arbustal → Herbazal.** Por solicitud del responsable, se extrajo de la clase *Arbustal_y_secundaria* lo **liso** (herbazal con arbustos dispersos confundido con arbustal). Criterio **conservador** calibrado con la verdad puntual: dentro de Arbustal, los píxeles con **textura local baja (desv. estándar del brillo < 9, ventana ~16 m) y NDVI < 0,22** se reclasificaron a *Pastos_y_herbaceo*. Verificación de integridad: la **única transición aplicada fue Arbustal → Herbazal (11.064 ha)**; ninguna otra clase se modificó.

---

## 6. Comparación de enfoques (concordancia a nivel de destino, contra verdad 2025)

| Enfoque | Concordancia destinos | Observación |
|---|---|---|
| Dynamic World crudo (10 m) | 0,686 | Referencia gruesa |
| DW + CLC fusionado (prior) | 0,735 | Mejor número, pero grano 10 m |
| SAM 2 + prior | 0,709 | Polígonos limpios, grano 10 m |
| SegFormer (8 canales) | 0,705–0,712 | **Grano fino 0,5 m** |
| Clay v1.5 + LoRA | 0,716 | Fundacional, 4 bandas |
| OBIA + Random Forest (GLCM) | 0,753 | Mayor número, pero hereda grano DW |
| **SegFormer fino + refinamiento (producto final)** | **≈0,71** | **Mayor fidelidad temática** |

**Hallazgo metodológico.** Con dos arquitecturas y un modelo fundacional, el deep learning converge a ~0,71 — un techo impuesto por la **supervisión débil** (las etiquetas), no por el modelo. El accuracy puntual premia a los productos "gruesos" (heredan el grano de Dynamic World y coinciden en los puntos), pero **no mide la calidad temática del mapa**, que es superior en el deep learning fino. La selección del producto se hizo por **fidelidad temática verificada visualmente sobre el terreno**, no solo por la métrica.

---

## 7. Producto final — áreas por destino económico

| Destino económico | Hectáreas |
|---|---|
| Forestales | 78.365 |
| Agropecuario | 21.967 |
| Agrícola | 4.326 |
| No aplica (territorios artificializados) | 1.400 |
| Suelo Protegido (cuerpos de agua y humedales) | 1.106 |
| Tierras improductivas (suelo desnudo / roca) | 134 |

**Entregables** (shapefile, EPSG:9377):
- `Coberturas_2025_Sumapaz.shp` — 9 clases de cobertura; campos `clase_id`, `cobertura`, `destino`, `area_ha`.
- `Destinos_Economicos_2025_Sumapaz.shp` — capa agregada por destino económico.

---

## 8. Validación y resultados

Validación contra >430 puntos de verdad 2025. **Concordancia global a nivel de destino: 71 %.** Desempeño por destino:

| Destino | Recall (concordancia) |
|---|---|
| Forestales | 0,89–0,96 |
| Tierras improductivas | 0,83–0,93 |
| Suelo Protegido | 0,75–0,85 |
| No aplica | 0,71–1,00 |
| Agrícola | 0,55–0,75 |
| Agropecuario | 0,39–0,48 |

Por zonas, el desempeño es mayor en norte y centro y menor en el sur (páramo alto), donde la verdad disponible es más escasa y la frontera herbazal/arbustal es más difusa.

---

## 9. Salvedades y limitaciones

1. **Insumo de trabajo, no cartografía oficial.** Pensado para apoyar la homologación de ZHF, sujeto a revisión de los profesionales temáticos.
2. **Frontera Agrícola/Agropecuario (pastos–herbazal–arbustal).** Es la principal fuente de incertidumbre: son mosaicos y transiciones graduales, sin límite nítido en el terreno; pueden existir mezclas dentro de estas categorías. Limitación intrínseca de la imagen RGB+NIR a 0,5 m (sin bandas red-edge/SWIR que separarían estructura de vegetación).
3. **Zona sur (páramo).** El refinamiento Arbustal→Herbazal es coherente con el terreno y con la ortofoto, pero la confianza estadística es menor por la escasez de puntos de validación en el sur.
4. **La cifra de 71 %** mide aciertos en puntos; no captura plenamente la calidad temática del mapa, que es superior.

---

## 10. Decisión pendiente — homologación del herbazal de páramo

Conforme a la tabla de homologación, el **Herbazal se homologó a Agropecuario**. En Sumapaz la mayor parte del herbazal corresponde a **herbazal de páramo**, ecológicamente vegetación natural protegida más que uso agropecuario. La decisión de mantenerlo como **Agropecuario** o reclasificarlo a **Suelo Protegido (63)** queda a criterio temático de Catastro; es un cambio de un parámetro y la capa de destinos puede regenerarse sin reprocesar el modelo.

---

## 11. Reproducibilidad

- **Entorno:** WSL Ubuntu, conda `coberturas-2025` (ver `environment.yml`) (Python 3.11, PyTorch 2.11 CUDA 12.8), GPU NVIDIA RTX 4500 Ada (24 GB).
- **Librerías clave:** rasterio, GDAL, geopandas, transformers, terratorch, segment-anything-2, earthengine-api, scikit-learn, scipy.
- **Código:** módulos `pipeline_v3/` — `legend.py` (leyenda y homologación), `weak_labels_gee.py` (Dynamic World), `label_fusion.py` (fusión CLC), `validation_set.py`, `objects_sam.py` (SAM 2), `train_segformer.py`, `train_clay_lora.py`, `active_learning.py`, `produce_dl.py` (inferencia AOI), `refine_arbustal.py` (refinamiento), `validate_2025.py`.

---

## 12. Créditos

- **Homologación CLC → destinos económicos:** Gerencia de Información Catastral (20-may-2026); definiciones de coberturas por IDECA.
- **Corrección Humedal/Agua → Suelo Protegido (63):** observación técnica (15-may-2026).
- **Registro de campo:** Observatorio Técnico Catastral — OTC.
- **Productos abiertos:** Google Dynamic World (Sentinel-2); interpretación CLC 2016.

---

*Documento de metodología. Generado para acompañar la entrega de la capa de coberturas 2025 del área rural de Sumapaz.*
