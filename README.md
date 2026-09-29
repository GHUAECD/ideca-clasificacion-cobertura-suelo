# IDECA - Clasificación de coberturas del suelo

## Descripción

El proyecto clasifica coberturas de la tierra 2025 en la zona rural de Sumapaz, Bogotá D.C., y las homologa a destinos económicos catastrales.

## Metodología

Exploración no supervisada (K-Means / OBIA) → clasificación supervisada con referencia CLC 2016 (Random Forest y SegFormer) → clasificación con referencias 2025 y SegFormer como modelo seleccionado → validación independiente con puntos 2025 → homologación a destinos económicos.

La etapa final empleó Dynamic World 2025 y CLC 2016 únicamente para coberturas estables, con ajustes posteriores de arbustal a pastos y herbáceo. El aprendizaje activo se evaluó con Clay + LoRA como experimento y no se incorporó al modelo SegFormer entregado.

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
- La validación no es probabilística. La documentación advierte una posible sobrestimación de la mejora posterior al refinamiento si se usaron los mismos puntos para calibrarlo y validarlo.

## Estado del código

El código fuente está disponible en [coberturas-rurales-2025/](coberturas-rurales-2025/README.md), versión 3.0.0. Incluye el flujo final con SegFormer, módulos experimentales, tablas de configuración y pruebas básicas; de las etapas anteriores se conservan solo los módulos reutilizados por el flujo final.

La instalación documentada usa Conda en Linux / WSL2 y una GPU NVIDIA para entrenamiento e inferencia en tiempos razonables. Desde la raíz de este repositorio:

```bash
cd coberturas-rurales-2025
conda env create -f environment.yml
conda activate coberturas-2025
```

Las dependencias están en [environment.yml](coberturas-rurales-2025/environment.yml) y la secuencia de ejecución en [ejecutar_flujo_final.sh](coberturas-rurales-2025/scripts/ejecutar_flujo_final.sh). Los comandos deben ejecutarse desde `coberturas-rurales-2025/`, con los insumos preparados según su [guía de datos](coberturas-rurales-2025/README.md#datos). La descarga de Dynamic World requiere autenticación en Earth Engine y un proyecto habilitado.

La reproducción completa requiere datos y pesos entrenados no incluidos, además de pasos manuales de etiquetado y validación. El script de calibración del refinamiento no se conservó; consulte las [limitaciones documentadas](coberturas-rurales-2025/README.md#limitaciones-y-problemas-conocidos).

## Documentación

La [presentación técnica del proyecto](Presentaci%C3%B3n%20Coberturas%20Rural%202026.pdf) es el documento de soporte del ejercicio.

El código incorpora la [metodología](coberturas-rurales-2025/docs/METODOLOGIA.md), la [documentación técnica](coberturas-rurales-2025/docs/DOCUMENTACION_TECNICA.md) y el [historial de versiones](coberturas-rurales-2025/CHANGELOG.md).

## Entidad

Unidad Administrativa Especial de Catastro Distrital - UAECD  
Infraestructura de Datos Espaciales para el Distrito Capital - IDECA
