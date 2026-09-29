"""Pipeline v3: coberturas 2025 orientadas a destinos económicos catastrales (Sumapaz).

Reemplaza el núcleo defectuoso de v1/v2 (etiqueta CLC-2016 desfasada, validación
circular, leyenda de 10 clases con pares inseparables) por:

- Leyenda orientada a los 6 destinos económicos (``legend.py``).
- Etiquetas débiles contemporáneas 2025 desde GEE (``weak_labels_gee.py``).
- Capa de objetos con SAM 2 (``objects_sam.py``).
- Active learning con un set pequeño auto-etiquetado (``active_learning.py``).
- Validación contra verdad 2025 a nivel de destino (``validate_2025.py``).

Reutiliza el andamiaje de v1/v2: preparación de orto/MDT, ``feature_stack``,
inferencia por ventanas y post-proceso (sieve + UMC).
"""
