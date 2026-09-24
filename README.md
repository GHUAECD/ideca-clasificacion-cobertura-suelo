# IDECA - Clasificación de coberturas del suelo

## Descripción

El proyecto clasifica coberturas de la tierra 2025 en la zona rural de Sumapaz, Bogotá D.C., y las homologa a destinos económicos catastrales.

## Metodología

Exploración no supervisada (K-Means / OBIA) → clasificación supervisada con referencia CLC 2016 (Random Forest y SegFormer) → clasificación con referencias 2025 y SegFormer como modelo seleccionado → validación independiente con puntos 2025 → homologación a destinos económicos.

La etapa final empleó Dynamic World 2025 y CLC 2016 únicamente para coberturas estables, con aprendizaje activo y ajustes posteriores sobre herbazal.

## Clases de cobertura

1. Construido
2. Cultivos
3. Pastos y herbáceo
4. Bosque
5. Arbustal y secundaria
6. Frailejonal
7. Suelo desnudo / roca
8. Humedal
9. Agua

## Resultados principales

- Aproximadamente 71 % de concordancia a nivel de destino económico frente a puntos independientes verificados en 2025.
- 9 clases de cobertura.
- 6 destinos económicos.
- Ortoimagen 2025 de 0,5 m.
- Productos en EPSG:9377 - MAGNA-SIRGAS 2018 / Origen Nacional.

## Productos

- `Coberturas_2025_Sumapaz`
- `Destinos_Economicos_2025_Sumapaz`

Los archivos geográficos pesados no se distribuyen directamente mediante GitHub y deben mantenerse en local.

## Limitaciones

- El producto es un insumo de trabajo y no constituye cartografía oficial.
- Existe mayor incertidumbre en la frontera agrícola/agropecuaria.
- Está sujeto a revisión temática.

## Estado del código

El código fuente utilizado para el procesamiento se encuentra pendiente de incorporación al repositorio.

El repositorio aún no permite reproducir completamente el procesamiento. Cuando se entregue el código, esta sección deberá actualizarse con las instrucciones de instalación, dependencias y ejecución.

## Documentación

La [presentación técnica del proyecto](Presentaci%C3%B3n%20Coberturas%20Rural%202026.pdf) es el documento de soporte del ejercicio.

## Entidad

Unidad Administrativa Especial de Catastro Distrital - UAECD  
Infraestructura de Datos Espaciales para el Distrito Capital - IDECA
