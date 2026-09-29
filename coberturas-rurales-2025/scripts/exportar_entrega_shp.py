"""Exporta el producto refinado a los shapefiles de entrega (último paso del flujo).

En junio de 2026 la conversión se hizo de forma interactiva; este script la deja
reproducible con el mismo esquema de la entrega:

- ``Coberturas_2025_Sumapaz.shp``: capa ``cobertura`` del GeoPackage refinado con los
  campos renombrados ``model_id -> clase_id`` y ``model_class -> cobertura``
  (se conservan ``destino`` y ``area_ha``).
- ``Destinos_Economicos_2025_Sumapaz.shp``: capa ``destinos`` (6 registros).

Codificación UTF-8 (archivo .cpg) y EPSG:9377.

Uso:
    python scripts/exportar_entrega_shp.py \
        --gpkg workspace_rural_aoi/deliverables/producto_dl_refinado/cobertura_2025_dl_refinado_destinos.gpkg \
        --out-dir workspace_rural_aoi/deliverables/entrega_shp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd


def main() -> None:
    ap = argparse.ArgumentParser(description="GeoPackage refinado -> shapefiles de entrega")
    ap.add_argument("--gpkg", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cob = gpd.read_file(args.gpkg, layer="cobertura")
    cob = cob.rename(columns={"model_id": "clase_id", "model_class": "cobertura"})
    cob = cob[["clase_id", "cobertura", "destino", "area_ha", "geometry"]]
    out_cob = args.out_dir / "Coberturas_2025_Sumapaz.shp"
    cob.to_file(out_cob, encoding="utf-8")
    print(f"{len(cob)} polígonos -> {out_cob}")

    dest = gpd.read_file(args.gpkg, layer="destinos")[["destino", "geometry"]]
    out_dest = args.out_dir / "Destinos_Economicos_2025_Sumapaz.shp"
    dest.to_file(out_dest, encoding="utf-8")
    print(f"{len(dest)} destinos -> {out_dest}")


if __name__ == "__main__":
    main()
