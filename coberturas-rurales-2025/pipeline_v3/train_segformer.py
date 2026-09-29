"""Entrenamiento baseline v3: SegFormer con etiquetas débiles DW 2025 (Fase 3a).

Diferencias clave frente a v2 (cuyo núcleo de código se reutiliza):
  - Etiquetas: Dynamic World 2025 homologado (contemporáneo a la orto), alineado por
    ventana al vuelo desde 10 m (nearest) — sin máscara de estabilidad 2016.
  - Leyenda: 9 clases orientadas a destinos (``legend.MODEL_LEGEND``).
  - Descuento de confianza al prior en "Bosque": el etiquetado del usuario demostró
    que DW infla 'trees' en el páramo (60 puntos prior Bosque → 21 confirmados), así
    que la clase 3 pesa menos en la pérdida (``--bosque-discount``).
  - Insumos definitivos: orto/MDT re-recortados al AOI efectivo.

Selección de modelo por macro-F1 contra las etiquetas débiles del split val (proxy);
la evaluación real contra los puntos del usuario la hace ``validate_2025.py``.

Uso (smoke):  python -m pipeline_v3.train_segformer --epochs 1 --limit-chips 200
Uso (full):   python -m pipeline_v3.train_segformer --epochs 12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoConfig, SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_v2.utils import align_single_band, feature_stack  # noqa: E402
from pipeline_v2.segformer_v2 import (  # noqa: E402
    DiceLoss,
    adapt_input_channels,
    fallback_segformer_config,
)
from pipeline_v3.legend import IGNORE_INDEX, num_model_classes  # noqa: E402
from pipeline_v3.prepare import read_label_window_aligned  # noqa: E402

WS = ROOT / "workspace_rural_aoi"
DEFAULTS = {
    "ortho": WS / "data/prepared/ortho_2025_aoi_efectivo.tif",
    "mdt": WS / "data/prepared/mdt_2025_aoi_efectivo.tif",
    "labels": WS / "data/labels/weak_labels_dw2025_model.tif",
    "chips": WS / "data/intermediate/v3_chips.csv",
    "out_dir": WS / "models/segformer_v3",
}
FEATURES = {"elevation_min": 2400.0, "elevation_max": 4200.0, "slope_scale": 1.0}
IN_CHANNELS = 8
BOSQUE_CLASS_ID = 3


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeakLabelDataset(Dataset):
    """Chips 512px: orto 0,5 m + MDT alineado + etiquetas DW 10 m alineadas al vuelo."""

    def __init__(self, args, split: str) -> None:
        self.args = args
        self.split = split
        frame = pd.read_csv(args.chips)
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if args.limit_chips and split == "train":
            frame = frame.sample(n=min(args.limit_chips, len(frame)), random_state=42).reset_index(drop=True)
        if args.limit_chips and split == "val":
            frame = frame.sample(n=min(max(args.limit_chips // 4, 20), len(frame)), random_state=42).reset_index(drop=True)
        self.frame = frame
        self.ortho = self.mdt = self.labels = None

    def __len__(self) -> int:
        return len(self.frame)

    def _open(self) -> None:
        if self.ortho is None:
            self.ortho = rasterio.open(self.args.ortho)
            self.mdt = rasterio.open(self.args.mdt)
            self.labels = rasterio.open(self.args.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._open()
        row = self.frame.iloc[idx]
        win = Window(int(row["col_off"]), int(row["row_off"]), int(row["size_px"]), int(row["size_px"]))
        ortho = self.ortho.read([1, 2, 3, 4], window=win)
        elev = align_single_band(self.mdt, self.ortho, win)
        x = feature_stack(
            ortho, elev,
            pixel_size=float(self.ortho.res[0]),
            elevation_min=FEATURES["elevation_min"],
            elevation_max=FEATURES["elevation_max"],
            slope_scale=FEATURES["slope_scale"],
        )
        y = read_label_window_aligned(self.labels, self.ortho, win).astype(np.int64)
        # fuera de la imagen (collar negro) no se aprende nada
        y[np.all(ortho == 0, axis=0)] = IGNORE_INDEX

        if self.split == "train":
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy(); y = y[:, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy(); y = y[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy(); y = np.rot90(y, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.35:
                gain = np.random.uniform(0.92, 1.08)
                bias = np.random.uniform(-0.025, 0.025)
                x[:4] = np.clip(x[:4] * gain + bias, 0.0, 1.0)
        return {"pixel_values": torch.from_numpy(x), "labels": torch.from_numpy(y)}


def build_model(hf_model: str, num_labels: int) -> SegformerForSemanticSegmentation:
    try:
        cfg = AutoConfig.from_pretrained(hf_model, num_labels=num_labels)
        model = SegformerForSemanticSegmentation.from_pretrained(
            hf_model, config=cfg, ignore_mismatched_sizes=True)
        print(f"backbone preentrenado: {hf_model}")
    except Exception as exc:
        print(f"sin backbone preentrenado ({exc}); usando config fallback")
        model = SegformerForSemanticSegmentation(fallback_segformer_config(num_labels, IN_CHANNELS))
    adapt_input_channels(model, IN_CHANNELS)
    return model


def class_weights_from_chips(chips_csv: Path, num_labels: int, bosque_discount: float) -> torch.Tensor:
    stats = pd.read_csv(chips_csv)
    train = stats[stats["split"] == "train"]
    counts = np.array([train[f"class_{i}_px10m"].sum() for i in range(num_labels)], dtype=np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = np.clip(weights / weights.mean(), 0.2, 5.0)
    weights[BOSQUE_CLASS_ID] *= bosque_discount  # prior DW infla Bosque en páramo
    return torch.tensor(weights, dtype=torch.float32)


def sampler_weights(frame: pd.DataFrame, num_labels: int) -> np.ndarray:
    counts = np.array([max(int((frame["dominant_class"] == c).sum()), 1) for c in range(num_labels)])
    inv = 1.0 / counts
    w = frame["dominant_class"].map(lambda c: inv[int(c)] if c >= 0 else inv.min()).to_numpy()
    return w / w.sum()


def update_confusion(cm, target, pred, num_labels, stride=4):
    t = target[::stride, ::stride].ravel()
    p = pred[::stride, ::stride].ravel()
    ok = t != IGNORE_INDEX
    if ok.any():
        idx = t[ok] * num_labels + p[ok]
        cm += np.bincount(idx, minlength=num_labels * num_labels).reshape(num_labels, num_labels)


def macro_f1(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
    return float(np.mean(f1))


def main() -> None:
    ap = argparse.ArgumentParser(description="SegFormer baseline v3 (etiquetas DW 2025)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=Path, default=v)
    ap.add_argument("--hf-model", default="nvidia/mit-b2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--bosque-discount", type=float, default=0.5)
    ap.add_argument("--limit-chips", type=int, default=0, help="submuestreo para smoke test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = num_model_classes()

    train_ds = WeakLabelDataset(args, "train")
    val_ds = WeakLabelDataset(args, "val")
    print(f"chips train: {len(train_ds)} | val: {len(val_ds)} | device: {device}")

    sw = sampler_weights(train_ds.frame, num_labels)
    sampler = WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(args.hf_model, num_labels).to(device)
    cw = class_weights_from_chips(args.chips, num_labels, args.bosque_discount).to(device)
    print("pesos de clase:", [round(float(w), 2) for w in cw])
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=IGNORE_INDEX)
    dice = DiceLoss()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler(device)

    warmup = max(1, args.epochs // 6)
    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []

    for epoch in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_factor(epoch)
        model.train()
        t0, tr_loss, n_batches = time.time(), 0.0, 0
        for batch in train_dl:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(pixel_values=x).logits
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = ce(logits, y) + 0.5 * dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss.detach())
            n_batches += 1

        model.eval()
        cm = np.zeros((num_labels, num_labels), dtype=np.int64)
        vl_loss, v_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["pixel_values"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                with autocast(device):
                    logits = model(pixel_values=x).logits
                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    vl_loss += float(ce(logits, y))
                v_batches += 1
                pred = logits.argmax(1).cpu().numpy()
                update_confusion(cm, batch["labels"].numpy(), pred, num_labels)

        f1 = macro_f1(cm)
        rec = {
            "epoch": epoch + 1,
            "lr": opt.param_groups[0]["lr"],
            "train_loss": tr_loss / max(n_batches, 1),
            "val_loss": vl_loss / max(v_batches, 1),
            "val_macro_f1_weak": f1,
            "secs": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_macro_f1_weak": f1},
                       out_dir / "best.pt")
            np.save(out_dir / "best_confusion_weak.npy", cm)

    (out_dir / "training_log.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                    "best_val_macro_f1_weak": best_f1, "history": history}, indent=2),
        encoding="utf-8")
    print(f"mejor val macro-F1 (vs etiquetas débiles): {best_f1:.4f} -> {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
