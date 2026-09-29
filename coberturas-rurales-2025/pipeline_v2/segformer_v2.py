from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoConfig, SegformerConfig, SegformerForSemanticSegmentation

from .config import V2Config
from .utils import (
    IGNORE_INDEX,
    align_single_band,
    append_jsonl,
    cosine_weight,
    ensure_dir,
    ensure_parent,
    feature_stack,
    write_json,
)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_chip_stats(config: V2Config) -> dict:
    paths = config.paths
    chips = pd.read_csv(paths["chips"])
    rows = []
    num_labels = int(config.classes["num_labels"])
    with rasterio.open(paths["labels"]) as label_src, rasterio.open(paths["train_mask"]) as mask_src:
        for _, row in chips.iterrows():
            win = Window(int(row["col_off"]), int(row["row_off"]), int(row["width"]), int(row["height"]))
            labels = label_src.read(1, window=win)
            if row["split"] == "train":
                valid = mask_src.read(1, window=win).astype(bool) & (labels != IGNORE_INDEX)
            else:
                valid = labels != IGNORE_INDEX
            counts = np.bincount(labels[valid].astype(np.int32), minlength=num_labels) if valid.any() else np.zeros(num_labels)
            rec = row.to_dict()
            for class_id in range(num_labels):
                rec[f"class_{class_id}_px"] = int(counts[class_id])
            rec["valid_px"] = int(counts.sum())
            rec["dominant_class"] = int(np.argmax(counts)) if counts.sum() else -1
            rows.append(rec)
    stats = pd.DataFrame(rows)

    train = stats[(stats["split"] == "train") & (stats["valid_px"] > 0)]
    totals = np.array([train[f"class_{i}_px"].sum() for i in range(num_labels)], dtype=np.float64)
    totals[totals == 0] = 1.0
    inv = totals.sum() / totals
    inv = inv / inv.mean()
    hard_boost = {int(k): float(v) for k, v in config.classes.get("hard_class_boost", {}).items()}

    weights = []
    for _, row in stats.iterrows():
        if row["split"] != "train" or row["valid_px"] <= 0:
            weights.append(0.0)
            continue
        counts = np.array([row[f"class_{i}_px"] for i in range(num_labels)], dtype=np.float64)
        frac = counts / max(counts.sum(), 1.0)
        weight = float((frac * inv).sum())
        for class_id, boost in hard_boost.items():
            if counts[class_id] > 0:
                weight *= boost
        weights.append(weight)
    stats["sample_weight"] = weights
    ensure_parent(paths["chips_v2"])
    stats.to_csv(paths["chips_v2"], index=False, encoding="utf-8")
    summary = {
        "chips_v2": str(paths["chips_v2"]),
        "chips_by_split": stats.groupby("split").size().to_dict(),
        "train_class_pixels": {str(i): int(totals[i]) for i in range(num_labels)},
    }
    write_json(summary, config.wpath("logs", "chips_v2_stats_summary.json"))
    return summary


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_safe = target.clone()
        target_safe[~valid] = 0
        one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid
        inter = (probs * one_hot).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3)) + 1e-6
        dice = (2.0 * inter + 1e-6) / denom
        return 1.0 - dice.mean()


class FocalCELoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = float(gamma)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index, reduction="none")
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        pt = torch.exp(-ce[valid])
        focal = ((1.0 - pt) ** self.gamma) * ce[valid]
        return focal.mean()


class CoverageV2Dataset(Dataset):
    def __init__(self, config: V2Config, split: str) -> None:
        self.config = config
        self.split = split
        stats_path = config.paths["chips_v2"] if config.paths["chips_v2"].exists() else config.paths["chips"]
        self.frame = pd.read_csv(stats_path)
        self.frame = self.frame[self.frame["split"] == split].reset_index(drop=True)
        if "valid_px" in self.frame.columns:
            self.frame = self.frame[self.frame["valid_px"] > 0].reset_index(drop=True)
        self.ortho = None
        self.mdt = None
        self.labels = None
        self.mask = None

    def __len__(self) -> int:
        return len(self.frame)

    def _open(self) -> None:
        if self.ortho is None:
            self.ortho = rasterio.open(self.config.paths["ortho"])
        if self.mdt is None:
            self.mdt = rasterio.open(self.config.paths["mdt"])
        if self.labels is None:
            self.labels = rasterio.open(self.config.paths["labels"])
        if self.mask is None:
            self.mask = rasterio.open(self.config.paths["train_mask"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._open()
        row = self.frame.iloc[idx]
        win = Window(int(row["col_off"]), int(row["row_off"]), int(row["width"]), int(row["height"]))
        ortho = self.ortho.read([1, 2, 3, 4], window=win)
        elev = align_single_band(self.mdt, self.ortho, win)
        x = feature_stack(
            ortho,
            elev,
            pixel_size=float(self.ortho.res[0]),
            elevation_min=float(self.config.features["elevation_min"]),
            elevation_max=float(self.config.features["elevation_max"]),
            slope_scale=float(self.config.features["slope_scale"]),
        )
        y = self.labels.read(1, window=win).astype(np.int64)
        if self.split == "train":
            valid = self.mask.read(1, window=win).astype(bool)
            y[~valid] = IGNORE_INDEX
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.35:
                gain = np.random.uniform(0.92, 1.08)
                bias = np.random.uniform(-0.025, 0.025)
                x[:4] = np.clip(x[:4] * gain + bias, 0.0, 1.0)
        return {"pixel_values": torch.from_numpy(x), "labels": torch.from_numpy(y)}


def fallback_segformer_config(num_labels: int, in_channels: int) -> SegformerConfig:
    return SegformerConfig(
        num_channels=in_channels,
        num_labels=num_labels,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        reshape_last_stage=True,
    )


def adapt_input_channels(model: SegformerForSemanticSegmentation, in_channels: int) -> None:
    proj = model.segformer.encoder.patch_embeddings[0].proj
    if proj.in_channels == in_channels:
        return
    new_proj = nn.Conv2d(
        in_channels=in_channels,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=proj.bias is not None,
    )
    with torch.no_grad():
        repeat = math.ceil(in_channels / proj.in_channels)
        expanded = proj.weight.repeat(1, repeat, 1, 1)[:, :in_channels]
        expanded[:, : proj.in_channels] = proj.weight
        if in_channels > proj.in_channels:
            expanded[:, proj.in_channels :] = proj.weight.mean(dim=1, keepdim=True)
        new_proj.weight.copy_(expanded)
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)
    model.segformer.encoder.patch_embeddings[0].proj = new_proj
    model.config.num_channels = in_channels


def build_model(config: V2Config) -> SegformerForSemanticSegmentation:
    num_labels = int(config.classes["num_labels"])
    in_channels = int(config.features["in_channels"])
    model_name = str(config.paths["hf_model"])
    try:
        hf_cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            config=hf_cfg,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    except Exception:
        model = SegformerForSemanticSegmentation(fallback_segformer_config(num_labels, in_channels))
    adapt_input_channels(model, in_channels)
    return model


def class_weights(config: V2Config) -> torch.Tensor:
    stats = pd.read_csv(config.paths["chips_v2"])
    train = stats[stats["split"] == "train"]
    num_labels = int(config.classes["num_labels"])
    counts = np.array([train[f"class_{i}_px"].sum() for i in range(num_labels)], dtype=np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    for key, boost in config.classes.get("hard_class_boost", {}).items():
        weights[int(key)] *= float(boost)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class Metrics:
    loss: float
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    macro_recall: float


def metrics_from_confusion(cm: np.ndarray) -> dict:
    total = float(cm.sum())
    if total == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "macro_recall": 0.0}
    diag = np.diag(cm).astype(np.float64)
    row_sum = cm.sum(axis=1).astype(np.float64)
    col_sum = cm.sum(axis=0).astype(np.float64)
    recall = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)
    precision = np.divide(diag, col_sum, out=np.zeros_like(diag), where=col_sum > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(diag), where=(precision + recall) > 0)
    present = row_sum > 0
    return {
        "accuracy": float(diag.sum() / total),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
    }


def update_confusion(cm: np.ndarray, target: np.ndarray, pred: np.ndarray, num_labels: int, stride: int = 1) -> None:
    if stride > 1:
        target = target[::stride, ::stride]
        pred = pred[::stride, ::stride]
    valid = target != IGNORE_INDEX
    if not valid.any():
        return
    idx = target[valid].astype(np.int64) * num_labels + pred[valid].astype(np.int64)
    binc = np.bincount(idx, minlength=num_labels * num_labels)
    cm += binc.reshape(num_labels, num_labels)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    scaler: GradScaler,
    ce_loss: nn.Module,
    dice_loss: DiceLoss,
    config: V2Config,
    device: torch.device,
) -> Metrics:
    train = optimizer is not None
    model.train(train)
    num_labels = int(config.classes["num_labels"])
    cm = np.zeros((num_labels, num_labels), dtype=np.int64)
    total_loss = 0.0
    grad_accum = int(config.training["gradient_accumulation"])
    amp_enabled = bool(config.training["amp"]) and torch.cuda.is_available()
    ce_w = float(config.training["ce_weight"])
    dice_w = float(config.training["dice_weight"])
    val_stride = 1 if train else int(config.training.get("val_pixel_stride", 1))

    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(pixel_values=x).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = ce_w * ce_loss(logits, y) + dice_w * dice_loss(logits, y)
            if train:
                loss = loss / grad_accum
        if train:
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * (grad_accum if train else 1)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        target = y.detach().cpu().numpy().astype(np.uint8)
        for b in range(pred.shape[0]):
            update_confusion(cm, target[b], pred[b], num_labels, stride=val_stride)
    vals = metrics_from_confusion(cm)
    return Metrics(
        loss=total_loss / max(len(loader), 1),
        macro_f1=vals["macro_f1"],
        accuracy=vals["accuracy"],
        balanced_accuracy=vals["balanced_accuracy"],
        macro_recall=vals["macro_recall"],
    )


def train(config: V2Config) -> dict:
    seed_everything(config.seed)
    if not config.paths["chips_v2"].exists():
        build_chip_stats(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CoverageV2Dataset(config, "train")
    val_ds = CoverageV2Dataset(config, "val")
    stats = pd.read_csv(config.paths["chips_v2"])
    train_stats = stats[(stats["split"] == "train") & (stats["valid_px"] > 0)]
    max_samples = int(config.training.get("max_train_chips_per_epoch", 0))
    if bool(config.training.get("weighted_sampler", True)):
        weights = train_stats["sample_weight"].clip(lower=1e-6).to_numpy(dtype=np.float64)
        num_samples = min(max_samples, len(weights)) if max_samples else len(weights)
        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.training["batch_size"]),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=int(config.training["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.training["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.training["batch_size"]),
        shuffle=False,
        num_workers=int(config.training["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.training["num_workers"]) > 0,
    )

    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config.training["learning_rate"]), weight_decay=float(config.training["weight_decay"]))
    weights = class_weights(config).to(device)
    ce_loss = FocalCELoss(weights, gamma=float(config.training["focal_gamma"]))
    dice_loss = DiceLoss()
    scaler = GradScaler(enabled=bool(config.training["amp"]) and torch.cuda.is_available())
    epochs = int(config.training["epochs"])
    warmup = int(config.training["warmup_epochs"])
    early = int(config.training["early_stopping"])
    best_f1 = -1.0
    patience = 0
    history = []
    ensure_dir(config.paths["model_dir"])

    def lr_for_epoch(epoch: int) -> float:
        base = float(config.training["learning_rate"])
        if epoch < warmup:
            return base * float(epoch + 1) / max(warmup, 1)
        t = (epoch - warmup) / max(epochs - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * t))

    for epoch in range(epochs):
        start = time.time()
        lr = lr_for_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_m = run_epoch(model, train_loader, optimizer, scaler, ce_loss, dice_loss, config, device)
        val_m = run_epoch(model, val_loader, None, scaler, ce_loss, dice_loss, config, device)
        rec = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_m.loss,
            "train_macro_f1": train_m.macro_f1,
            "val_loss": val_m.loss,
            "val_macro_f1": val_m.macro_f1,
            "val_balanced_accuracy": val_m.balanced_accuracy,
            "seconds": time.time() - start,
        }
        history.append(rec)
        append_jsonl(rec, config.wpath("logs", "segformer_v2_training.jsonl"))
        if val_m.macro_f1 > best_f1:
            best_f1 = val_m.macro_f1
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "history": history,
                    "best_val_macro_f1": best_f1,
                    "config": config.raw,
                },
                config.paths["best_model"],
            )
        else:
            patience += 1
            if patience >= early:
                break
    result = {
        "device": str(device),
        "best_val_macro_f1": best_f1,
        "epochs_completed": len(history),
        "checkpoint": str(config.paths["best_model"]),
        "history": history,
    }
    write_json(result, config.paths["train_metrics"])
    return result


def chip_positions(length: int, chip: int, stride: int) -> list[int]:
    if length <= chip:
        return [0]
    pos = list(range(0, length - chip + 1, stride))
    if pos[-1] != length - chip:
        pos.append(length - chip)
    return pos


def infer(config: V2Config) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.paths["best_model"], map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    num_labels = int(config.classes["num_labels"])
    tile = int(config.inference["tile_size_px"])
    halo = int(config.inference["halo_px"])
    chip = int(config.inference["chip_size_px"])
    stride = int(config.inference["stride_px"])
    batch_size = int(config.inference["batch_size"])
    amp_enabled = bool(config.inference["amp"]) and torch.cuda.is_available()
    weight = cosine_weight(chip)

    with rasterio.open(config.paths["ortho"]) as ortho_src, rasterio.open(config.paths["mdt"]) as mdt_src:
        pred_profile = ortho_src.profile.copy()
        pred_profile.update(count=1, dtype=rasterio.uint8, nodata=IGNORE_INDEX, compress="lzw", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="YES")
        unc_profile = ortho_src.profile.copy()
        unc_profile.update(count=1, dtype=rasterio.float32, nodata=1.0, compress="lzw", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="YES")
        ensure_parent(config.paths["prediction"])
        with rasterio.open(config.paths["prediction"], "w", **pred_profile) as pred_dst, rasterio.open(config.paths["uncertainty"], "w", **unc_profile) as unc_dst:
            tile_id = 0
            for row_off in range(0, ortho_src.height, tile):
                for col_off in range(0, ortho_src.width, tile):
                    core_h = min(tile, ortho_src.height - row_off)
                    core_w = min(tile, ortho_src.width - col_off)
                    read_row = max(0, row_off - halo)
                    read_col = max(0, col_off - halo)
                    read_h = min(ortho_src.height - read_row, core_h + (row_off - read_row) + halo)
                    read_w = min(ortho_src.width - read_col, core_w + (col_off - read_col) + halo)
                    win = Window(read_col, read_row, read_w, read_h)
                    ortho = ortho_src.read([1, 2, 3, 4], window=win)
                    elev = align_single_band(mdt_src, ortho_src, win)
                    stack = feature_stack(
                        ortho,
                        elev,
                        pixel_size=float(ortho_src.res[0]),
                        elevation_min=float(config.features["elevation_min"]),
                        elevation_max=float(config.features["elevation_max"]),
                        slope_scale=float(config.features["slope_scale"]),
                    )
                    h, w = stack.shape[-2:]
                    logits_sum = np.zeros((num_labels, h, w), dtype=np.float32)
                    weight_sum = np.zeros((h, w), dtype=np.float32)
                    jobs = [(r, c) for r in chip_positions(h, chip, stride) for c in chip_positions(w, chip, stride)]
                    for start in range(0, len(jobs), batch_size):
                        batch_jobs = jobs[start : start + batch_size]
                        batch = np.stack([stack[:, r : r + chip, c : c + chip] for r, c in batch_jobs])
                        tensor = torch.from_numpy(batch).to(device)
                        with torch.no_grad(), autocast(enabled=amp_enabled):
                            logits = model(pixel_values=tensor).logits
                            logits = F.interpolate(logits, size=(chip, chip), mode="bilinear", align_corners=False)
                        logits_np = logits.detach().cpu().numpy().astype(np.float32)
                        for arr, (r, c) in zip(logits_np, batch_jobs):
                            logits_sum[:, r : r + chip, c : c + chip] += arr * weight[None]
                            weight_sum[r : r + chip, c : c + chip] += weight
                    logits_avg = logits_sum / np.maximum(weight_sum[None], 1e-6)
                    logits_avg -= logits_avg.max(axis=0, keepdims=True)
                    probs = np.exp(logits_avg)
                    probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-6)
                    pred = probs.argmax(axis=0).astype(np.uint8)
                    top1 = probs.max(axis=0).astype(np.float32)
                    uncertainty = 1.0 - top1
                    core_r = row_off - read_row
                    core_c = col_off - read_col
                    out_win = Window(col_off, row_off, core_w, core_h)
                    pred_dst.write(pred[core_r : core_r + core_h, core_c : core_c + core_w], 1, window=out_win)
                    unc_dst.write(uncertainty[core_r : core_r + core_h, core_c : core_c + core_w], 1, window=out_win)
                    tile_id += 1
                    append_jsonl({"tile_id": tile_id, "row_off": row_off, "col_off": col_off, "jobs": len(jobs)}, config.paths["progress"])
    return {"prediction": str(config.paths["prediction"]), "uncertainty": str(config.paths["uncertainty"]), "checkpoint": str(config.paths["best_model"])}
