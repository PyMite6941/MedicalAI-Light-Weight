import argparse
import csv
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

DATA_DIR = "./data"
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fusion_model.pth")
CONFIDENCE_THRESHOLD = 0.75

CSV_COLUMNS = ["image_path", "source", "symptoms", "diagnosis", "labels"]

def load_label_list():
    if not os.path.exists(CSV_PATH):
        return []
    labels = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row.get("diagnosis", "").strip().lower()
            if d:
                labels.add(d)
    return sorted(labels)

def prepare_data():
    from datasets import load_dataset
    os.makedirs(IMAGES_DIR, exist_ok=True)

    file_exists = os.path.exists(CSV_PATH)
    if not file_exists:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

    rows_written = 0
    sources_done = []

    # ── IU-Xray (image + question + report) ──
    print("Downloading IU-Xray from Hugging Face...")
    iuxray = load_dataset("ayyuce/Indiana_University_Chest_X-ray_Collection", split="train")
    written = 0
    for i, example in enumerate(iuxray):
        symptoms = (example.get("question") or "").strip()
        diagnosis = (example.get("report") or "").strip()
        image = example.get("image")
        if not symptoms or not diagnosis or image is None:
            continue
        image_path = os.path.join(IMAGES_DIR, f"iu_xray_{i}.jpg")
        image.convert("RGB").save(image_path)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([image_path, "iu_xray", symptoms, diagnosis, ""])
        written += 1
    print(f"  IU-Xray: {written} rows")
    rows_written += written
    sources_done.append(f"iu_xray ({written})")

    # ── NIH Chest X-ray (image + disease labels) ──
    print("Downloading NIH Chest X-ray from Hugging Face...")
    nih = load_dataset("g-ronimo/NIH-Chest-X-ray-dataset_resized300px", split="train", streaming=True)
    label_names = [
        "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
        "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]
    written = 0
    for i, example in enumerate(nih):
        if written >= 3000:
            break
        image = example.get("image")
        label_indices = example.get("labels", [])
        if image is None or not label_indices:
            continue
        label_str = "|".join(label_names[idx] for idx in label_indices)
        primary_diagnosis = label_names[label_indices[0]]
        image_path = os.path.join(IMAGES_DIR, f"nih_{i}.jpg")
        image.convert("RGB").save(image_path)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([image_path, "nih", "", primary_diagnosis, label_str])
        written += 1
        if written % 500 == 0:
            print(f"  NIH progress: {written}...")
    print(f"  NIH: {written} rows")
    rows_written += written
    sources_done.append(f"nih ({written})")

    print(f"Done. Total: {rows_written} rows written to {CSV_PATH}")
    print(f"Sources: {', '.join(sources_done)}")

def add_data(image_path, symptoms, diagnosis, labels=""):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([image_path, "user", symptoms, diagnosis, labels])
    print(f"Added 1 row to {CSV_PATH}: diagnosis='{diagnosis}'")

class FusionDataset(Dataset):
    def __init__(self, csv_path, label_list):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = row.get("diagnosis", "").strip().lower()
                if d:
                    self.rows.append(row)
        self.label_list = label_list

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        symptoms = row.get("symptoms", "").strip()
        label_idx = self.label_list.index(row["diagnosis"].strip().lower())
        return image, symptoms, label_idx

def collate_fn(batch):
    images = [item[0] for item in batch]
    symptoms = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, symptoms, labels

class DiagnosisFusionModel(nn.Module):
    def __init__(self, num_conditions):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.symptom_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.symptom_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.symptom_encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_conditions),
        )

    def encode_images(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            return self.image_encoder.get_image_features(**inputs)

    def encode_symptoms(self, symptom_texts):
        inputs = self.symptom_tokenizer(
            symptom_texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        with torch.no_grad():
            outputs = self.symptom_encoder(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    def forward(self, images, symptom_texts):
        image_vecs = self.encode_images(images)
        symptom_vecs = self.encode_symptoms(symptom_texts)
        combined = torch.cat([image_vecs, symptom_vecs], dim=-1)
        return self.classifier(combined)

def train(epochs, batch_size, lr, val_split, use_amp, grad_accum):
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    _console = Console()

    has_gpu = torch.cuda.is_available()
    use_amp = use_amp and has_gpu
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    label_list = load_label_list()
    if not label_list:
        _console.print("[red]No data found. Run --mode prepare-data or --mode add-data first.[/red]")
        return

    dataset = FusionDataset(CSV_PATH, label_list)
    val_size = max(int(val_split * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    _console.print(f"[bold cyan]Training Setup[/bold cyan]")
    _console.print(f"  Classes: {len(label_list)}")
    _console.print(f"  Train/Val: {len(train_subset)}/{len(val_subset)}")
    _console.print(f"  Batch size: {batch_size}  Grad accum: {grad_accum}")
    _console.print(f"  Device: {'GPU' if has_gpu else 'CPU'}  AMP: {'ON' if use_amp else 'OFF'}")

    model = DiagnosisFusionModel(num_conditions=len(label_list))
    if has_gpu:
        model = model.cuda()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        _console.print(f"\n[bold yellow]Epoch {epoch + 1}/{epochs}[/bold yellow]")
        _console.print("-" * 40)

        # ── Train ──
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        train_progress = Progress(
            TextColumn("[cyan]  Train[/cyan]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[green]{task.fields[loss]:.4f}[/green]"),
            TimeElapsedColumn(),
            transient=True,
        )
        with train_progress:
            task = train_progress.add_task("", total=len(train_loader), loss=0.0)
            for i, (images, symptoms, labels) in enumerate(train_loader):
                if has_gpu:
                    labels = labels.cuda()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images, symptoms)
                    loss = loss_fn(logits, labels)
                loss = loss / grad_accum
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * grad_accum
                train_progress.update(task, advance=1, loss=loss.item() * grad_accum)

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, symptoms, labels in val_loader:
                if has_gpu:
                    labels = labels.cuda()
                logits = model(images, symptoms)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Train loss", f"{avg_train_loss:.4f}")
        table.add_row("Val loss",   f"{avg_val_loss:.4f}")
        _console.print(table)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({"model_state": model.classifier.state_dict(), "label_list": label_list}, CHECKPOINT_PATH)
    _console.print(f"[green]Saved checkpoint to {CHECKPOINT_PATH}[/green]")

def test(batch_size):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found. Run --mode train first.")
        return
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    label_list = checkpoint["label_list"]
    model = DiagnosisFusionModel(num_conditions=len(label_list))
    model.classifier.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = FusionDataset(CSV_PATH, label_list)
    test_size = max(int(0.2 * len(dataset)), 1)
    _, test_subset = random_split(dataset, [len(dataset) - test_size, test_size])
    loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    correct = 0
    inconclusive = 0
    total = 0
    with torch.no_grad():
        for images, symptoms, labels in loader:
            logits = model(images, symptoms)
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)
            for i in range(len(labels)):
                total += 1
                if confidence[i].item() < CONFIDENCE_THRESHOLD:
                    inconclusive += 1
                elif predicted[i].item() == labels[i].item():
                    correct += 1
    print(f"Tested on {total} held-out examples")
    print(f"Correct (above confidence threshold): {correct} ({100 * correct / total:.1f}%)")
    print(f"Flagged as inconclusive / needs follow-up: {inconclusive} ({100 * inconclusive / total:.1f}%)")

def info():
    if not os.path.exists(CSV_PATH):
        print("No dataset.csv found. Run --mode prepare-data first.")
        return
    sources = {}
    total = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = row.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            total += 1
    print(f"Dataset: {CSV_PATH}")
    print(f"Total rows: {total}")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    print(f"Images dir: {IMAGES_DIR}")
    img_count = len([x for x in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, x))]) if os.path.exists(IMAGES_DIR) else 0
    print(f"Images: {img_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test the medical image+symptom fusion model")
    parser.add_argument("--mode", required=True, choices=["prepare-data", "add-data", "train", "test", "info"])
    parser.add_argument("--image", help="Path to an image file (for --mode add-data)")
    parser.add_argument("--symptoms", help="Symptom description text (for --mode add-data)")
    parser.add_argument("--diagnosis", help="Diagnosis label (for --mode add-data)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision (GPU only)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    if args.mode == "prepare-data":
        prepare_data()
    elif args.mode == "add-data":
        if not (args.image and args.symptoms and args.diagnosis):
            print("--mode add-data requires --image, --symptoms, and --diagnosis")
        else:
            add_data(args.image, args.symptoms, args.diagnosis)
    elif args.mode == "train":
        train(args.epochs, args.batch_size, args.lr, args.val_split, args.use_amp, args.grad_accum)
    elif args.mode == "test":
        test(args.batch_size)
    elif args.mode == "info":
        info()
