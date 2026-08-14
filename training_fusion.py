import argparse
import csv
import json
import os
import random
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from optimize import get_device, set_cpu_threads

DATA_DIR = "./data"
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fusion_model.pth")
LABELS_PATH = os.path.join(CHECKPOINT_DIR, "labels.json")
CONFIDENCE_THRESHOLD = 0.75
CSV_COLUMNS = ["image_path", "source", "symptoms", "diagnosis", "labels"]

CANONICAL_LABELS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
    "Calcified_Granuloma", "Granulomatous_Disease", "Bronchiectasis",
    "Hiatal_Hernia", "Pneumoperitoneum", "Pericardial_Effusion",
    "Rib_Fracture", "COPD", "Interstitial_Lung_Disease", "Sarcoidosis",
    "Lung_Cancer", "Metastatic_Disease", "Mediastinal_Mass",
    "Aortic_Aneurysm", "Aortic_Dissection",
    "Pulmonary_Embolism", "Pulmonary_Hypertension",
    "Tuberculosis", "Aspergillosis",
    "ARDS", "Pulmonary_Edema", "Alveolar_Hemorrhage",
    "Organizing_Pneumonia", "Hypersensitivity_Pneumonitis",
    "Lymphangitic_Carcinomatosis", "Lymphadenopathy",
    "Pleural_Plaque", "Pneumoconiosis", "Silicosis", "Asbestosis",
    "Empyema", "Hemothorax", "Chylothorax",
    "Round_Atelectasis", "Mucus_Plugging", "Tree_in_Bud",
    "Cavitary_Lesion", "Cystic_Lung_Disease",
    "Post_Surgical_Changes", "Support_Devices",
    "Pancoast_Tumor", "Diaphragmatic_Hernia",
    "Eventration", "Subcutaneous_Emphysema",
    "Ankylosing_Spondylitis", "Kyphoscoliosis",
    "Thoracic_Spine_Fracture",
]

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}
NUM_CLASSES = len(CANONICAL_LABELS)

SYMPTOM_TEMPLATES = [
    "What abnormality is present in this chest X-ray?",
    "Patient presents with shortness of breath and cough.",
    "Routine pre-operative chest X-ray.",
    "Patient with history of smoking. Evaluate for lung pathology.",
    "Fever and productive cough for {days} days. Assess for pneumonia.",
    "Chest pain and dyspnea on exertion.",
    "Post-surgical follow-up. Evaluate lung expansion.",
    "Patient with known COPD. Assess for acute changes.",
    "Trauma patient after MVA. Evaluate for pneumothorax.",
    "Immunocompromised patient with fever.",
    "Patient with weight loss and night sweats.",
    "Pre-employment screening chest X-ray.",
    "CHF follow-up. Evaluate for pulmonary edema.",
    "Hemoptysis for 2 weeks. Evaluate for bronchiectasis or mass.",
    "Contact TB patient. Screening chest X-ray.",
    "HIV positive patient with cough and fever.",
    "Post-chemotherapy evaluation. Neutropenic fever.",
    "Rule out TB in patient with positive PPD.",
    "Dyspnea and orthopnea. Evaluate for heart failure.",
    "Suspected pulmonary embolism.",
    "Chronic cough with mucus production.",
    "Rheumatoid arthritis patient with new dyspnea.",
    "Patient with asbestos exposure history.",
    "Pre-renal transplant evaluation.",
    "ARDS follow-up chest X-ray.",
    "Ventilator-associated pneumonia surveillance.",
]

NIH_NORMALIZED_LABELS = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Effusion", "Infiltration": "Infiltration",
    "Mass": "Mass", "Nodule": "Nodule", "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax", "Consolidation": "Consolidation",
    "Edema": "Edema", "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis", "Pleural_Thickening": "Pleural_Thickening",
    "Hernia": "Hernia", "No Finding": "No Finding",
}


def parse_labels(label_str):
    if not label_str or label_str == "nan":
        return []
    parts = label_str.replace("|", ",").split(",")
    seen = set()
    result = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            result.append(p)
            seen.add(p)
    return result


def map_to_canonical(label_name):
    """Map a raw label string to its canonical class index.

    Exact matches always win. Only when there's no exact match do we fall
    back to substring matching, and then we prefer the *longest* (most
    specific) key — otherwise a broad label like "Effusion" would shadow
    a more specific one like "Pericardial_Effusion" simply because it
    appears earlier in CANONICAL_LABELS.
    """
    n = label_name.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = {key: key.lower().replace(" ", "_").replace("-", "_") for key in LABEL_TO_IDX}

    for key, key_norm in normalized.items():
        if n == key_norm:
            return LABEL_TO_IDX[key]

    best_key, best_len = None, -1
    for key, key_norm in normalized.items():
        if (n in key_norm or key_norm in n) and len(key_norm) > best_len:
            best_key, best_len = key, len(key_norm)
    return LABEL_TO_IDX[best_key] if best_key else -1


def labels_to_multihot(label_names):
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for name in label_names:
        idx = map_to_canonical(name)
        if idx >= 0:
            vec[idx] = 1.0
    return vec


def load_label_list():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    if not os.path.exists(CSV_PATH):
        return []
    labels = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_str = row.get("labels", "").strip()
            names = parse_labels(label_str)
            if not names:
                d = row.get("diagnosis", "").strip().lower()
                if d:
                    names = [d]
            for n in names:
                idx = map_to_canonical(n)
                if idx >= 0:
                    labels.add(CANONICAL_LABELS[idx])
    return sorted(labels) if labels else CANONICAL_LABELS


def write_csv_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def append_row(image_path, source, symptoms, diagnosis, labels_str):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([image_path, source, symptoms, diagnosis, labels_str])


def prepare_data():
    write_csv_header()
    rows_written = 0

    rows_written += _load_iuxray()
    rows_written += _load_nih_full()
    rows_written += _load_chexpert()

    print(f"Done. Total: {rows_written} rows written to {CSV_PATH}")


def _load_iuxray():
    from datasets import load_dataset
    print("Downloading IU-Xray from Hugging Face...")
    try:
        ds = load_dataset("ayyuce/Indiana_University_Chest_X-ray_Collection", split="train", trust_remote_code=True)
    except Exception:
        print("  IU-Xray not available, skipping.")
        return 0

    written = 0
    for i, example in enumerate(ds):
        symptoms = (example.get("question") or "").strip()
        diagnosis = (example.get("report") or "").strip()
        image = example.get("image")
        if not diagnosis or image is None:
            continue
        image_path = os.path.join(IMAGES_DIR, f"iu_xray_{i}.jpg")
        image.convert("RGB").save(image_path)
        append_row(image_path, "iu_xray", symptoms, diagnosis, diagnosis)
        written += 1
        if written % 1000 == 0:
            print(f"  IU-Xray progress: {written}...")
    print(f"  IU-Xray: {written} rows")
    return written


def _load_nih_full():
    from datasets import load_dataset
    print("Downloading NIH ChestX-ray14 from Hugging Face (streaming, full dataset)...")
    try:
        ds = load_dataset("g-ronimo/NIH-Chest-X-ray-dataset_resized300px", split="train", streaming=True, trust_remote_code=True)
    except Exception:
        print("  NIH dataset not available, skipping.")
        return 0

    written = 0
    max_images = 50000
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        label_indices = example.get("labels", [])
        if image is None or not label_indices:
            continue
        label_names = [NIH_NORMALIZED_LABELS.get(
            ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
             "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
             "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"][idx], "")
            for idx in label_indices]
        label_names = [l for l in label_names if l]
        if not label_names:
            continue
        labels_str = "|".join(label_names)
        primary_diagnosis = label_names[0]
        image_path = os.path.join(IMAGES_DIR, f"nih_{i}.jpg")
        image.convert("RGB").save(image_path)
        append_row(image_path, "nih", "", primary_diagnosis, labels_str)
        written += 1
        if written % 5000 == 0:
            print(f"  NIH progress: {written}...")
    print(f"  NIH ChestX-ray14: {written} rows")
    return written


def _load_chexpert():
    print("Downloading CheXpert from Hugging Face (subset, ~30K images)...")
    from datasets import load_dataset
    try:
        ds = load_dataset("StanfordAIMI/chexpert", split="train", streaming=True, trust_remote_code=True)
    except Exception:
        print("  CheXpert not available via HF, trying alternative...")
        return _load_chexpert_alt()

    written = 0
    chexpert_labels = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices",
    ]
    chexpert_map = {
        "No Finding": "No Finding", "Cardiomegaly": "Cardiomegaly",
        "Edema": "Edema", "Consolidation": "Consolidation",
        "Atelectasis": "Atelectasis", "Pneumothorax": "Pneumothorax",
        "Pleural Effusion": "Effusion", "Fracture": "Rib_Fracture",
        "Lung Opacity": "Infiltration", "Lung Lesion": "Nodule",
        "Pneumonia": "Pneumonia", "Support Devices": "Support_Devices",
    }
    for i, example in enumerate(ds):
        if written >= 30000:
            break
        image = example.get("image")
        if image is None:
            continue
        found_labels = []
        for cl in chexpert_labels:
            val = example.get(cl, 0)
            if val == 1:
                mapped = chexpert_map.get(cl)
                if mapped:
                    found_labels.append(mapped)
        if not found_labels:
            continue
        labels_str = "|".join(found_labels)
        image_path = os.path.join(IMAGES_DIR, f"chexpert_{i}.jpg")
        image.convert("RGB").save(image_path)
        append_row(image_path, "chexpert", "", found_labels[0], labels_str)
        written += 1
        if written % 5000 == 0:
            print(f"  CheXpert progress: {written}...")
    print(f"  CheXpert: {written} rows")
    return written


def _load_chexpert_alt():
    print("  CheXpert alternative download not implemented. Skipping.")
    return 0


def add_data(image_path, symptoms, diagnosis, labels=""):
    write_csv_header()
    append_row(image_path, "user", symptoms, diagnosis, labels)
    print(f"Added 1 row: diagnosis='{diagnosis}'")


class FusionDataset(Dataset):
    def __init__(self, csv_path, label_list, mode="multilabel"):
        self.mode = mode
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = row.get("image_path", "").strip()
                if p and os.path.exists(p):
                    self.rows.append(row)
        self.label_list = label_list if label_list else CANONICAL_LABELS

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=128)

        symptoms = row.get("symptoms", "").strip()
        if not symptoms:
            tmpl = random.choice(SYMPTOM_TEMPLATES)
            symptoms = tmpl.format(days=random.choice([3, 5, 7, 10, 14]))

        if self.mode == "multilabel":
            label_str = row.get("labels", "").strip()
            names = parse_labels(label_str)
            if not names:
                d = row.get("diagnosis", "").strip()
                if d:
                    names = [d]
            label = labels_to_multihot(names)
        else:
            diagnosis = row.get("diagnosis", "").strip().lower()
            label = self.label_list.index(diagnosis) if diagnosis in self.label_list else 0
            label = torch.tensor(label, dtype=torch.long)

        return image, symptoms, label


def collate_multilabel(batch):
    images = [item[0] for item in batch]
    symptoms = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    return images, symptoms, labels


def collate_singlelabel(batch):
    images = [item[0] for item in batch]
    symptoms = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, symptoms, labels


class DiagnosisFusionModel(nn.Module):
    def __init__(self, num_conditions, freeze_encoders=True, wider_head=False):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.symptom_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.symptom_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        if freeze_encoders:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.symptom_encoder.parameters():
                param.requires_grad = False
        if wider_head:
            self.classifier = nn.Sequential(
                nn.Linear(512 + 768, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_conditions),
            )
        else:
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


def train(epochs, batch_size, lr, val_split, use_amp, grad_accum, mode, wider_head):
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    _console = Console()

    set_cpu_threads()
    accel = get_device()
    if accel not in ("cuda", "mps"):
        accel = "cpu"  # TPU training needs a torch_xla-specific loop; not implemented here.
    device = torch.device(accel)
    has_gpu = accel != "cpu"
    use_amp = use_amp and accel == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    label_list = load_label_list()
    if not label_list:
        _console.print("[red]No data found. Run --mode prepare-data first.[/red]")
        return

    dataset = FusionDataset(CSV_PATH, label_list, mode=mode)
    val_size = max(int(val_split * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    collate = collate_multilabel if mode == "multilabel" else collate_singlelabel
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)

    num_classes = len(label_list) if mode == "singlelabel" else NUM_CLASSES
    _console.print(f"[bold cyan]Training Setup[/bold cyan]")
    _console.print(f"  Mode: {mode}  Classes: {num_classes}")
    _console.print(f"  Train/Val: {len(train_subset)}/{len(val_subset)}")
    _console.print(f"  Batch: {batch_size}  Grad accum: {grad_accum}")
    _console.print(f"  Device: {accel.upper()}  AMP: {'ON' if use_amp else 'OFF'}  Head: {'wide' if wider_head else 'std'}")

    model = DiagnosisFusionModel(num_classes, freeze_encoders=True, wider_head=wider_head)
    if has_gpu:
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if mode == "multilabel":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        _console.print(f"\n[bold yellow]Epoch {epoch + 1}/{epochs}[/bold yellow]")

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        train_progress = Progress(
            TextColumn("[cyan]  Train[/cyan]"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[green]{task.fields[loss]:.4f}[/green]"),
            TimeElapsedColumn(), transient=True,
        )
        with train_progress:
            task = train_progress.add_task("", total=len(train_loader), loss=0.0)
            for i, (images, symptoms, labels) in enumerate(train_loader):
                if has_gpu:
                    labels = labels.to(device)
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, symptoms, labels in val_loader:
                if has_gpu:
                    labels = labels.to(device)
                logits = model(images, symptoms)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Train loss", f"{avg_train_loss:.4f}")
        table.add_row("Val loss", f"{avg_val_loss:.4f}")
        table.add_row("LR", f"{scheduler.get_last_lr()[0]:.6f}")
        _console.print(table)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model_state": model.classifier.state_dict(),
        "label_list": label_list if mode == "singlelabel" else CANONICAL_LABELS,
        "num_classes": num_classes,
        "mode": mode,
    }, CHECKPOINT_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(CANONICAL_LABELS if mode == "multilabel" else label_list, f)
    _console.print(f"[green]Saved checkpoint to {CHECKPOINT_PATH}[/green]")


def test(batch_size, mode):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found. Run --mode train first.")
        return
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    label_list = checkpoint.get("label_list", CANONICAL_LABELS)
    mode = checkpoint.get("mode", mode)
    num_classes = len(label_list) if mode == "singlelabel" else NUM_CLASSES
    collate = collate_multilabel if mode == "multilabel" else collate_singlelabel

    model = DiagnosisFusionModel(num_classes)
    model.classifier.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = FusionDataset(CSV_PATH, label_list, mode=mode)
    test_size = max(int(0.2 * len(dataset)), 1)
    _, test_subset = random_split(dataset, [len(dataset) - test_size, test_size])
    loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    if mode == "multilabel":
        total = 0
        correct = 0
        inconclusive = 0
        with torch.no_grad():
            for images, symptoms, labels in loader:
                logits = model(images, symptoms)
                probs = torch.sigmoid(logits)
                for i in range(len(labels)):
                    total += 1
                    pred = (probs[i] >= 0.5).float()
                    confidence = probs[i].max().item()
                    if confidence < CONFIDENCE_THRESHOLD:
                        inconclusive += 1
                    elif torch.all(pred == labels[i].float()):
                        correct += 1
        print(f"Tested on {total} held-out examples (multi-label)")
        print(f"Exact match (above threshold): {correct} ({100 * correct / total:.1f}%)")
        print(f"Inconclusive: {inconclusive} ({100 * inconclusive / total:.1f}%)")
    else:
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
        print(f"Correct: {correct} ({100 * correct / total:.1f}%)")
        print(f"Inconclusive: {inconclusive} ({100 * inconclusive / total:.1f}%)")


def info():
    if not os.path.exists(CSV_PATH):
        print("No dataset.csv found.")
        return
    sources = {}
    total = 0
    label_counts = Counter()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = row.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            total += 1
            ls = row.get("labels", "").strip()
            for n in parse_labels(ls):
                label_counts[n] += 1
    print(f"Dataset: {CSV_PATH}")
    print(f"Total rows: {total}")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    img_count = len([x for x in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, x))]) if os.path.exists(IMAGES_DIR) else 0
    print(f"Images: {img_count}")
    print(f"Canonical classes: {NUM_CLASSES}")
    print(f"Unique label names in CSV: {len(label_counts)}")
    print("\nTop labels:")
    for label, count in label_counts.most_common(20):
        idx = map_to_canonical(label)
        mapped = CANONICAL_LABELS[idx] if idx >= 0 else "(unmapped)"
        print(f"  {mapped:30s} ({label:20s}) -> {count:5d} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedicalAI Fusion Model Training Pipeline")
    parser.add_argument("--mode", required=True, choices=["prepare-data", "add-data", "train", "test", "info"])
    parser.add_argument("--image", help="Image path (for --mode add-data)")
    parser.add_argument("--symptoms", help="Symptom text (for --mode add-data)")
    parser.add_argument("--diagnosis", help="Diagnosis (for --mode add-data)")
    parser.add_argument("--labels", default="", help="Pipe-delimited labels (for --mode add-data)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--classify_mode", choices=["singlelabel", "multilabel"], default="multilabel")
    parser.add_argument("--wider_head", action="store_true", default=False)
    args = parser.parse_args()

    if args.mode == "prepare-data":
        prepare_data()
    elif args.mode == "add-data":
        if not (args.image and args.symptoms and args.diagnosis):
            print("--mode add-data requires --image, --symptoms, and --diagnosis")
        else:
            add_data(args.image, args.symptoms, args.diagnosis, args.labels)
    elif args.mode == "train":
        train(args.epochs, args.batch_size, args.lr, args.val_split,
              args.use_amp, args.grad_accum, args.classify_mode, args.wider_head)
    elif args.mode == "test":
        test(args.batch_size, args.classify_mode)
    elif args.mode == "info":
        info()