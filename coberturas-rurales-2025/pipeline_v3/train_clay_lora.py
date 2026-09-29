"""Fase 3b — Clay (modelo fundacional EO) + LoRA, solo-óptico (RGB+NIR).

Mismo dato, mismas etiquetas fusionadas, misma validación que el SegFormer baseline,
para una comparación limpia. Diferencias de diseño impuestas por Clay:
  - Entrada: solo las 4 bandas ópticas (BLUE,GREEN,RED,NIR). Sin MDT (decisión del
    proyecto para esta comparación). El SegFormer usó 8 canales; esta asimetría se
    reporta explícitamente.
  - Resolución: chips de 512 px (256 m) submuestreados a 256 px — tamaño nativo del ViT
    de Clay.
  - Encoder Clay CONGELADO; solo se entrenan adaptadores LoRA (en to_qkv/to_out de los
    12 bloques de atención) + una cabeza de segmentación convolucional ligera.

Uso (smoke): python -m pipeline_v3.train_clay_lora --epochs 1 --limit-chips 200
Uso (full):  python -m pipeline_v3.train_clay_lora --epochs 12
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_v2.segformer_v2 import DiceLoss  # noqa: E402
from pipeline_v3.legend import IGNORE_INDEX, num_model_classes  # noqa: E402
from pipeline_v3.prepare import read_label_window_aligned  # noqa: E402
from pipeline_v3.train_segformer import (  # noqa: E402
    class_weights_from_chips, macro_f1, sampler_weights, seed_everything, update_confusion,
)

WS = ROOT / "workspace_rural_aoi"
CLAY_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW"]
INPUT_PX = 256
LORA_TARGETS = ["to_qkv", "to_out"]
DEFAULTS = {
    "ortho": WS / "data/prepared/ortho_2025_aoi_efectivo.tif",
    "labels": WS / "data/labels/weak_labels_fused.tif",
    "chips": WS / "data/intermediate/v3_chips.csv",
    "out_dir": WS / "models/clay_lora_v3",
}


class ClayOpticalDataset(Dataset):
    """RGB+NIR del chip (512px) submuestreado a 256, normalizado 0-1; etiqueta alineada."""

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
        self.ortho = self.labels = None

    def __len__(self) -> int:
        return len(self.frame)

    def _open(self) -> None:
        if self.ortho is None:
            self.ortho = rasterio.open(self.args.ortho)
            self.labels = rasterio.open(self.args.labels)

    def __getitem__(self, idx: int) -> dict:
        self._open()
        row = self.frame.iloc[idx]
        win = Window(int(row["col_off"]), int(row["row_off"]), int(row["size_px"]), int(row["size_px"]))
        bgrn = self.ortho.read([3, 2, 1, 4], window=win).astype(np.float32) / 255.0  # B,G,R,NIR
        y = read_label_window_aligned(self.labels, self.ortho, win).astype(np.int64)
        y[np.all(self.ortho.read([1, 2, 3, 4], window=win) == 0, axis=0)] = IGNORE_INDEX

        x = torch.from_numpy(bgrn)
        x = F.interpolate(x[None], size=(INPUT_PX, INPUT_PX), mode="bilinear", align_corners=False)[0]
        yt = torch.from_numpy(y)[None, None].float()
        yt = F.interpolate(yt, size=(INPUT_PX, INPUT_PX), mode="nearest")[0, 0].long()

        if self.split == "train":
            if np.random.rand() < 0.5:
                x = torch.flip(x, [2]); yt = torch.flip(yt, [1])
            if np.random.rand() < 0.5:
                x = torch.flip(x, [1]); yt = torch.flip(yt, [0])
        return {"pixel_values": x, "labels": yt}


class ClaySegmenter(nn.Module):
    """Clay congelado + LoRA, con cabeza conv ligera (token map -> 9 clases a 256px)."""

    def __init__(self, num_labels: int, lora_r: int = 16, lora_alpha: int = 32) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from terratorch.registry import BACKBONE_REGISTRY

        backbone = BACKBONE_REGISTRY.build("timm_clay_v1_base", pretrained=True, bands=CLAY_BANDS)
        for p in backbone.parameters():
            p.requires_grad = False
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGETS,
                         lora_dropout=0.05, bias="none")
        self.backbone = get_peft_model(backbone, cfg)
        self.embed_dim = 768
        self.head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, num_labels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        feat = out[-1] if isinstance(out, (list, tuple)) else out  # (B, 1+N, C)
        tokens = feat[:, 1:, :]                                     # drop cls
        b, n, c = tokens.shape
        s = int(math.sqrt(n))
        fmap = tokens.transpose(1, 2).reshape(b, c, s, s)          # (B,C,32,32)
        logits = self.head(fmap)
        return F.interpolate(logits, size=(INPUT_PX, INPUT_PX), mode="bilinear", align_corners=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clay + LoRA solo-óptico (Fase 3b)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=Path, default=v)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bosque-discount", type=float, default=0.5)
    ap.add_argument("--limit-chips", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boost-truth", type=float, default=1.0,
                    help="factor de sobre-muestreo para chips con verdad puntual (col has_truth)")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = num_model_classes()

    train_ds = ClayOpticalDataset(args, "train")
    val_ds = ClayOpticalDataset(args, "val")
    print(f"chips train: {len(train_ds)} | val: {len(val_ds)} | device: {device}", flush=True)

    sw = sampler_weights(train_ds.frame, num_labels)
    if args.boost_truth > 1.0 and "has_truth" in train_ds.frame.columns:
        boost = np.where(train_ds.frame["has_truth"].to_numpy() > 0, args.boost_truth, 1.0)
        sw = sw * boost
        sw = sw / sw.sum()
        print(f"boost verdad puntual x{args.boost_truth} en {int((boost>1).sum())} chips", flush=True)
    sampler = WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = ClaySegmenter(num_labels).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"params entrenables (LoRA+head): {trainable/1e6:.2f}M de {total/1e6:.1f}M "
          f"({100*trainable/total:.1f}%)", flush=True)

    cw = class_weights_from_chips(args.chips, num_labels, args.bosque_discount).to(device)
    ce = nn.CrossEntropyLoss(weight=cw, ignore_index=IGNORE_INDEX)
    dice = DiceLoss()
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = GradScaler(device)
    warmup = max(1, args.epochs // 6)

    def lr_factor(e):
        if e < warmup:
            return (e + 1) / warmup
        t = (e - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []

    for epoch in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_factor(epoch)
        model.train()
        t0, tr_loss, nb = time.time(), 0.0, 0
        for batch in train_dl:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(x)
                loss = ce(logits, y) + 0.5 * dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += float(loss.detach()); nb += 1

        model.eval()
        cm = np.zeros((num_labels, num_labels), dtype=np.int64)
        vl, vb = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["pixel_values"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                with autocast(device):
                    logits = model(x)
                    vl += float(ce(logits, y))
                vb += 1
                update_confusion(cm, batch["labels"].numpy(), logits.argmax(1).cpu().numpy(), num_labels)

        f1 = macro_f1(cm)
        rec = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"],
               "train_loss": tr_loss / max(nb, 1), "val_loss": vl / max(vb, 1),
               "val_macro_f1_weak": f1, "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_macro_f1_weak": f1},
                       out_dir / "best.pt")

    (out_dir / "training_log.json").write_text(
        json.dumps({"best_val_macro_f1_weak": best_f1, "history": history,
                    "note": "Clay+LoRA solo-optico RGB+NIR, sin MDT"}, indent=2), encoding="utf-8")
    print(f"mejor val macro-F1 (weak): {best_f1:.4f} -> {out_dir/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
