"""
Fabrisketch High-Accuracy Trainer (`ml/train.py` v2.1)
======================================================
Key Accuracy Upgrades:
  1. Saves `checkpoints/val_prediction_preview.png` after EVERY epoch so stopping
     early with Ctrl+C still gives you the latest visual preview immediately.
  2. Class-Weighted Cross-Entropy + Multi-Class Dice Loss for thin lines & corners.
  3. On-the-Fly 8x Geometric Augmentation + Cosine Annealing LR + `--resume`.
"""

import os
import glob
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from unet_model import EmbroideryDigitizerUNet


class ProceduralEmbroideryDataset(Dataset):
    def __init__(self, data_dir: str, img_size: int = 256, augment: bool = True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
        if not self.image_paths:
            raise FileNotFoundError(
                f"No training images found in '{data_dir}/images'. "
                "Run `py ml/zero_pes_dataset.py --samples 1500` first!"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        mask_path = os.path.join(self.data_dir, "masks", f"{stem}.png")
        angle_path = os.path.join(self.data_dir, "angles", f"{stem}.npy")
        corner_path = os.path.join(self.data_dir, "corners_and_edges", f"{stem}.png")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        angle = np.load(angle_path)
        a0 = cv2.resize(angle[0], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        a1 = cv2.resize(angle[1], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        corner = cv2.imread(corner_path, cv2.IMREAD_GRAYSCALE)
        corner = cv2.resize(corner, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
                corner = np.fliplr(corner).copy()
                a0 = np.fliplr(a0).copy()
                a1 = -np.fliplr(a1).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
                corner = np.flipud(corner).copy()
                a0 = np.flipud(a0).copy()
                a1 = -np.flipud(a1).copy()
            if random.random() < 0.5:
                img = np.rot90(img).copy()
                mask = np.rot90(mask).copy()
                corner = np.rot90(corner).copy()
                a0 = -np.rot90(a0).copy()
                a1 = -np.rot90(a1).copy()

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        angle_t = torch.from_numpy(np.stack([a0, a1], axis=0)).float()
        corner_t = torch.from_numpy(corner).unsqueeze(0).float() / 255.0

        return img_t, mask_t, angle_t, corner_t


def multi_class_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice_per_class = (2.0 * intersection + 1.0) / (cardinality + 1.0)
    weights = torch.tensor([0.1, 0.3, 0.3, 0.3], device=logits.device)
    return 1.0 - torch.sum(dice_per_class * weights)


def compute_loss(logits, pred_angle, pred_corner, gt_mask, gt_angle, gt_corner):
    class_weights = torch.tensor([0.25, 1.0, 2.5, 4.5], device=logits.device)
    loss_ce = F.cross_entropy(logits, gt_mask, weight=class_weights)
    loss_dice = multi_class_dice_loss(logits, gt_mask, num_classes=4)
    loss_corner = F.binary_cross_entropy(pred_corner.clamp(1e-5, 1.0 - 1e-5), gt_corner)

    fg_mask = (gt_mask > 0).float().unsqueeze(1)
    cos_sim = torch.sum(pred_angle * gt_angle, dim=1, keepdim=True)
    loss_angle = torch.sum((1.0 - cos_sim) * fg_mask) / (torch.sum(fg_mask) + 1e-5)

    return loss_ce + loss_dice + 0.5 * loss_corner + 0.5 * loss_angle


def compute_foreground_accuracy(logits: torch.Tensor, gt_mask: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    fg = gt_mask > 0
    if not torch.any(fg):
        return 1.0
    correct = (preds[fg] == gt_mask[fg]).float().mean().item()
    return correct * 100.0


def save_validation_preview(model, sample, device, out_path="checkpoints/val_prediction_preview.png"):
    model.eval()
    img_t, mask_t, _, _ = sample
    with torch.no_grad():
        logits, _, corner_prob = model(img_t.unsqueeze(0).to(device))
        pred_cls = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_corner = (corner_prob.squeeze().cpu().numpy() * 255).astype(np.uint8)

    palette = np.array([[245, 245, 245], [200, 90, 30], [20, 175, 235], [40, 200, 50]], dtype=np.uint8)
    in_bgr = cv2.cvtColor((img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    gt_vis = palette[mask_t.numpy()]
    pred_vis = palette[pred_cls]
    corner_vis = cv2.cvtColor(pred_corner, cv2.COLOR_GRAY2BGR)

    grid = np.hstack([in_bgr, gt_vis, pred_vis, corner_vis])
    cv2.imwrite(out_path, grid)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training High-Accuracy Fabrisketch U-Net on device: {device}")

    full_ds = ProceduralEmbroideryDataset(args.data_dir, img_size=args.img_size, augment=True)
    val_len = max(1, int(0.1 * len(full_ds)))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EmbroideryDigitizerUNet(in_channels=3, base_filters=32, num_classes=4).to(device)
    ckpt_path = "checkpoints/fabrisketch_unet.pth"
    if args.resume and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Resumed existing weights from {ckpt_path}!")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, angles, corners in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            angles, corners = angles.to(device), corners.to(device)

            optimizer.zero_grad()
            logits, pred_angle, pred_corner = model(imgs)
            loss = compute_loss(logits, pred_angle, pred_corner, masks, angles, corners)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_fg_acc = 0.0
        with torch.no_grad():
            for imgs, masks, angles, corners in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                angles, corners = angles.to(device), corners.to(device)
                logits, pred_angle, pred_corner = model(imgs)
                val_loss += compute_loss(logits, pred_angle, pred_corner, masks, angles, corners).item()
                val_fg_acc += compute_foreground_accuracy(logits, masks)

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        avg_acc = val_fg_acc / max(1, len(val_loader))
        cur_lr = scheduler.get_last_lr()[0]

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_path)
            save_validation_preview(model, val_ds[0], device)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | Stitch Class Accuracy: {avg_acc:.1f}% | LR: {cur_lr:.5f}"
        )

    print(f"Training complete! Best weights and preview saved in checkpoints/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./procedural_embroidery_dataset")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Continue training from existing checkpoint")
    args = parser.parse_args()
    train(args)