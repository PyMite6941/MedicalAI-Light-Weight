"""
Musculoskeletal (bone) X-ray fracture screening — a separate, experimental
model path alongside the chest X-ray pipeline (training.py / training_fusion.py).

Uses FracAtlas (Abedeen et al., Scientific Data 2023), a free, CC-BY-2.5
licensed dataset of 4,083 extremity radiographs (hand/leg/hip/shoulder)
with fracture and hardware annotations:
  https://huggingface.co/datasets/yh0701/FracAtlas_dataset

This intentionally does NOT touch the chest X-ray model, its checkpoints,
or its label space — separate data dir, separate checkpoint dir, separate
label set. It shares only the device-detection / CPU-thread-capping
infrastructure in optimize.py, since that should apply everywhere.

Status: scaffold. No pretrained weights ship with this repo — run
`--mode prepare-data` then `--mode train` to produce one, same as the
chest X-ray pipeline.
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from optimize import get_device, set_cpu_threads

DATA_DIR = "./data/bone"
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CHECKPOINT_DIR = "./checkpoints/bone_fracture"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "bone_fracture_model.pth")
CONFIDENCE_THRESHOLD = 0.6

CSV_COLUMNS = ["image_path", "fractured", "hardware", "body_part"]

# Multi-label output: does the model think a fracture or surgical
# hardware is present. Kept intentionally small and binary-flag-shaped —
# this is a screening aid, not a full musculoskeletal read.
BONE_LABELS = ["Fracture", "Hardware"]


def write_csv_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def prepare_data(max_samples=None):
    """Download FracAtlas via Hugging Face and materialize it as
    image files + a CSV, matching the pattern used for the chest X-ray
    datasets in training.py / expand_dataset.py."""
    from datasets import load_dataset

    write_csv_header()
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("Downloading FracAtlas from Hugging Face (yh0701/FracAtlas_dataset)...")
    try:
        ds = load_dataset("yh0701/FracAtlas_dataset", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  Could not load FracAtlas: {e}")
        print("  This dataset requires 'trust_remote_code=True' support in your 'datasets' version.")
        return

    written = 0
    for i, example in enumerate(ds):
        if max_samples and written >= max_samples:
            break
        image = example.get("image")
        if image is None:
            continue

        fractured = int(bool(example.get("fractured", 0)))
        hardware = int(bool(example.get("hardware", 0)))
        body_part = next(
            (part for part in ("hand", "leg", "hip", "shoulder", "mixed") if example.get(part)),
            "unknown",
        )

        image_path = os.path.join(IMAGES_DIR, f"fracatlas_{i}.jpg")
        image.convert("RGB").save(image_path)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([image_path, fractured, hardware, body_part])
        written += 1
        if written % 500 == 0:
            print(f"  progress: {written}...")

    print(f"Done. {written} rows written to {CSV_PATH}")


class BoneFractureDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = row.get("image_path", "").strip()
                if p and os.path.exists(p):
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=128)
        label = torch.tensor(
            [float(row.get("fractured", 0)), float(row.get("hardware", 0))],
            dtype=torch.float32,
        )
        return image, label


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return images, labels


class BoneFractureModel(nn.Module):
    """Frozen CLIP image encoder + a small classifier head — same
    lightweight shape as the chest X-ray fusion model's image path, just
    without the symptom-text branch (FracAtlas has no clinical text)."""

    def __init__(self, num_labels=len(BONE_LABELS)):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )

    def encode_images(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        device = next(self.image_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            return self.image_encoder.get_image_features(**inputs)

    def forward(self, images):
        features = self.encode_images(images)
        return self.classifier(features)


def train(epochs, batch_size, lr, val_split):
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    console = Console()

    set_cpu_threads()
    accel = get_device()
    if accel not in ("cuda", "mps"):
        accel = "cpu"  # TPU training needs a torch_xla-specific loop; not implemented here.
    device = torch.device(accel)

    if not os.path.exists(CSV_PATH):
        console.print("[red]No data found. Run --mode prepare-data first.[/red]")
        return

    dataset = BoneFractureDataset(CSV_PATH)
    if len(dataset) < 10:
        console.print("[red]Not enough data to train (need at least 10 rows).[/red]")
        return

    val_size = max(int(val_split * len(dataset)), 1)
    train_subset, val_subset = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    console.print("[bold cyan]Bone Fracture Training Setup[/bold cyan]")
    console.print(f"  Labels: {BONE_LABELS}")
    console.print(f"  Train/Val: {len(train_subset)}/{len(val_subset)}")
    console.print(f"  Device: {accel.upper()}")

    model = BoneFractureModel().to(device) if accel != "cpu" else BoneFractureModel()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        console.print(f"\n[bold yellow]Epoch {epoch + 1}/{epochs}[/bold yellow]")
        model.train()
        train_loss = 0.0
        progress = Progress(
            TextColumn("[cyan]  Train[/cyan]"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), transient=True,
        )
        with progress:
            task = progress.add_task("", total=len(train_loader))
            for images, labels in train_loader:
                if accel != "cpu":
                    labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress.update(task, advance=1)

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                if accel != "cpu":
                    labels = labels.to(device)
                logits = model(images)
                val_loss += loss_fn(logits, labels).item()
        avg_val_loss = val_loss / len(val_loader)

        table = Table(show_header=False, box=None)
        table.add_row("Train loss", f"{avg_train_loss:.4f}")
        table.add_row("Val loss", f"{avg_val_loss:.4f}")
        console.print(table)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({"model_state": model.classifier.state_dict(), "labels": BONE_LABELS}, CHECKPOINT_PATH)
    console.print(f"[green]Saved checkpoint to {CHECKPOINT_PATH}[/green]")


def info():
    if not os.path.exists(CSV_PATH):
        print("No bone X-ray dataset found. Run --mode prepare-data first.")
        return
    total = 0
    fractured = 0
    hardware = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            fractured += int(row.get("fractured", 0))
            hardware += int(row.get("hardware", 0))
    print(f"Dataset: {CSV_PATH}")
    print(f"Total rows: {total}")
    print(f"Fractured: {fractured} ({100 * fractured / total:.1f}%)" if total else "Fractured: 0")
    print(f"Hardware present: {hardware} ({100 * hardware / total:.1f}%)" if total else "Hardware: 0")
    print(f"Checkpoint trained: {'yes' if os.path.exists(CHECKPOINT_PATH) else 'no'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bone X-ray fracture screening (experimental, separate from the chest pipeline)")
    parser.add_argument("--mode", required=True, choices=["prepare-data", "train", "info"])
    parser.add_argument("--max_samples", type=int, default=None, help="Cap rows during prepare-data (omit for the full ~4K images)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()

    if args.mode == "prepare-data":
        prepare_data(args.max_samples)
    elif args.mode == "train":
        train(args.epochs, args.batch_size, args.lr, args.val_split)
    elif args.mode == "info":
        info()
