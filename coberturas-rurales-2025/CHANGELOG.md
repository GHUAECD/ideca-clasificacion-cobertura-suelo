# Historial de versiones

## 3.0.0 — publicación del código (septiembre de 2026)

Código del flujo que produjo la entrega del 14 de junio de 2026
(`Coberturas_2025_Sumapaz.shp` y `Destinos_Economicos_2025_Sumapaz.shp`).

Cambios respecto al código de trabajo, sin alterar la lógica de procesamiento:

- Se retiraron rutas locales, el identificador del proyecto de Earth Engine y nombres de personas
  de comentarios y documentación (reemplazados por marcadores o por el rol institucional).
- Se añadieron `scripts/recortar_insumos.py` y `scripts/exportar_entrega_shp.py`, que dejan como
  código dos pasos hechos de forma interactiva. Se verificaron contra la entrega: misma grilla del
  MDT, y shapefile de coberturas idéntico byte a byte con 51.586 polígonos.
- Se añadieron `scripts/ejecutar_flujo_final.sh`, `environment.yml`, pruebas rápidas en `tests/`
  y la documentación en `docs/`.
- Del código de las etapas 1 y 2 solo se incluyen los módulos que el flujo final reutiliza.

## Etapas anteriores (no publicadas en este repositorio)

- **Etapa 2 (abril–mayo de 2026):** Random Forest y SegFormer entrenados con la interpretación
  Corine Land Cover 2016 (10 macroclases); entrega del 4 de mayo de 2026.
- **Etapa 1 (abril de 2026):** clasificación no supervisada exploratoria (K-Means y OBIA).
