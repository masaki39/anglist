"""
Minimal training loop for sagittal landmark heatmaps.
Expect data_dir to contain *_image.npy and *_landmarks.json exported from Slicer.
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HeatmapDataset, LANDMARK_ORDER
from model import SmallUNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Folder with *_image.npy and *_landmarks.json")
    p.add_argument("--save-dir", default="runs", help="Directory to save checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for heatmaps (pixels)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        img = batch["image"].to(device)
        target = batch["heatmap"].to(device)
        pred = model(img)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            img = batch["image"].to(device)
            target = batch["heatmap"].to(device)
            pred = model(img)
            loss = torch.mean((pred - target) ** 2)
            total_loss += loss.item() * img.size(0)
    return total_loss / len(loader.dataset)


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = HeatmapDataset(
        data_dir=args.data_dir,
        resize=tuple(args.resize),
        sigma=args.sigma,
    )
    # simple split: 90/10
    n_total = len(dataset)
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallUNet(num_landmarks=len(LANDMARK_ORDER)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"[{epoch}/{args.epochs}] train {train_loss:.4f} | val {val_loss:.4f}")
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  -> saved best.pt (val {val_loss:.4f})")


if __name__ == "__main__":
    main()
