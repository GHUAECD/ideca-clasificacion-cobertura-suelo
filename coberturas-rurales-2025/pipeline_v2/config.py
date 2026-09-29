from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V2Config:
    raw: dict[str, Any]
    config_path: Path

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def source_v1(self) -> dict[str, Any]:
        return self.raw["source_v1"]

    @property
    def workspace(self) -> dict[str, Any]:
        return self.raw["workspace"]

    @property
    def classes(self) -> dict[str, Any]:
        return self.raw["classes"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def inference(self) -> dict[str, Any]:
        return self.raw["inference"]

    @property
    def postprocess(self) -> dict[str, Any]:
        return self.raw["postprocess"]

    @property
    def unsupervised(self) -> dict[str, Any]:
        return self.raw.get("unsupervised", {})

    @property
    def unsupervised_obia(self) -> dict[str, Any]:
        return self.raw.get("unsupervised_obia", {})

    @property
    def workspace_root(self) -> Path:
        return Path(self.workspace["root"]).expanduser()

    @property
    def v1_root(self) -> Path:
        return Path(self.source_v1["workspace_root"]).expanduser()

    def wpath(self, *parts: str) -> Path:
        return self.workspace_root.joinpath(*parts)

    @property
    def paths(self) -> dict[str, Path]:
        v1 = self.v1_root
        return {
            "v1_aoi": v1 / "data" / "aoi" / "AOI_BOGOTA_RURAL.shp",
            "v1_ortho": v1 / "data" / "prepared" / "ortho_2025_aoi.tif",
            "v1_mdt": v1 / "data" / "prepared" / "mdt_2025_aoi.tif",
            "v1_labels": v1 / "data" / "labels" / "label_2016_macroclasses.tif",
            "v1_train_mask": v1 / "data" / "labels" / "train_mask_2016_stable.tif",
            "v1_chips": v1 / "data" / "intermediate" / "chips.csv",
            "v1_blocks": v1 / "data" / "intermediate" / "block_split.gpkg",
            "v1_mapping": v1 / "config" / "class_mapping.csv",
            "hf_model": Path(self.source_v1["hf_model"]).expanduser(),
            "ortho": self.wpath("data", "prepared", "ortho_2025_aoi.tif"),
            "mdt": self.wpath("data", "prepared", "mdt_2025_aoi.tif"),
            "labels": self.wpath("data", "labels", "label_2016_macroclasses.tif"),
            "train_mask": self.wpath("data", "labels", "train_mask_2016_stable.tif"),
            "chips": self.wpath("data", "intermediate", "chips.csv"),
            "chips_v2": self.wpath("data", "intermediate", "chips_v2_stats.csv"),
            "blocks": self.wpath("data", "intermediate", "block_split.gpkg"),
            "mapping": self.wpath("config", "class_mapping.csv"),
            "aoi": self.wpath("data", "aoi", "AOI_BOGOTA_RURAL.shp"),
            "model_dir": self.wpath("models", "segformer_v2"),
            "best_model": self.wpath("models", "segformer_v2", "best.pt"),
            "train_metrics": self.wpath("models", "segformer_v2", "training_metrics.json"),
            "prediction": self.wpath("inference", "segformer_v2_prediction.tif"),
            "uncertainty": self.wpath("inference", "segformer_v2_uncertainty.tif"),
            "final_raw_aoi": self.wpath("deliverables", "cobertura_2025_macroclases_v2_raw_aoi.tif"),
            "final_raster": self.wpath("deliverables", "cobertura_2025_macroclases_v2.tif"),
            "final_confidence_preview": self.wpath("deliverables", "cobertura_2025_confianza_v2_preview.tif"),
            "area_stats_csv": self.wpath("deliverables", "cobertura_2025_macroclases_v2_area.csv"),
            "area_stats_json": self.wpath("deliverables", "cobertura_2025_macroclases_v2_area.json"),
            "preview_png": self.wpath("deliverables", "preview_cobertura_2025_v2.png"),
            "postprocess_summary": self.wpath("deliverables", "postprocess_v2_summary.json"),
            "unsupervised_raw": self.wpath("deliverables", "unsupervised_enriched_kmeans25_10m_raw.tif"),
            "unsupervised_smooth": self.wpath("deliverables", "unsupervised_enriched_kmeans25_10m.tif"),
            "unsupervised_preview": self.wpath("deliverables", "preview_unsupervised_enriched_kmeans25_10m.png"),
            "unsupervised_area_csv": self.wpath("deliverables", "unsupervised_enriched_kmeans25_10m_area.csv"),
            "unsupervised_area_json": self.wpath("deliverables", "unsupervised_enriched_kmeans25_10m_area.json"),
            "unsupervised_interpretation_csv": self.wpath("deliverables", "unsupervised_enriched_kmeans25_10m_interpretation.csv"),
            "unsupervised_metrics": self.wpath("deliverables", "unsupervised_enriched_kmeans25_10m_metrics.json"),
            "unsupervised_report_md": self.wpath("deliverables", "informe_clasificacion_no_supervisada_enriquecida.md"),
            "obia_segments": self.wpath("deliverables", "unsupervised_obia_featurebins_segments_10m.tif"),
            "obia_raw": self.wpath("deliverables", "unsupervised_obia_featurebins_k10_10m_raw.tif"),
            "obia_final": self.wpath("deliverables", "unsupervised_obia_featurebins_k10_10m.tif"),
            "obia_vector": self.wpath("deliverables", "unsupervised_obia_featurebins_k10_10m.gpkg"),
            "obia_preview": self.wpath("deliverables", "preview_unsupervised_obia_featurebins_k10_10m.png"),
            "obia_area_csv": self.wpath("deliverables", "unsupervised_obia_featurebins_k10_10m_area.csv"),
            "obia_area_json": self.wpath("deliverables", "unsupervised_obia_featurebins_k10_10m_area.json"),
            "obia_interpretation_csv": self.wpath("deliverables", "unsupervised_obia_featurebins_k10_10m_interpretation.csv"),
            "obia_metrics": self.wpath("deliverables", "unsupervised_obia_featurebins_k10_10m_metrics.json"),
            "obia_report_md": self.wpath("deliverables", "informe_clasificacion_no_supervisada_obia_v3.md"),
            "progress": self.wpath("logs", "segformer_v2_progress.jsonl"),
            "stage_summary": self.wpath("logs", "stage_inputs_summary.json"),
        }


def load_config(path: str | Path) -> V2Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return V2Config(raw=raw, config_path=config_path)
