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
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fusion_model.pt")
CONFIDENCE_THRESHOLD = 0.75

def load_label_list():
    if not os.path.exists(CSV_PATH):
        return []
    labels = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.add(row["diagnosis"].strip().lower())
    return sorted(labels)

def prepare_data():
    from datasets import load_dataset
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print("Downloading IU-Xray dataset from Hugging Face (first run only)...")
    ds = load_dataset("ayyuce/Indiana_University_Chest_X-ray_Collection", split="train")
    rows_written = 0
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "symptoms", "diagnosis"])
        for i, example in enumerate(ds):
            symptoms = (example.get("indication") or "").strip()
            diagnosis = (example.get("impression") or "").strip()
            image = example.get("image")
            if not symptoms or not diagnosis or image is None:
                continue
            image_path = os.path.join(IMAGES_DIR, f"xray_{i}.jpg")
            image.convert("RGB").save(image_path)
            writer.writerow([image_path, symptoms, diagnosis])
            rows_written += 1
    print(f"Done. Wrote {rows_written} image+symptom+diagnosis rows to {CSV_PATH}")

def add_data(image_path, symptoms, diagnosis):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path", "symptoms", "diagnosis"])
        writer.writerow([image_path, symptoms, diagnosis])
    print(f"Added 1 row to {CSV_PATH}: diagnosis='{diagnosis}'")

class FusionDataset(Dataset):
    def __init__(self, csv_path, label_list):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        self.label_list = label_list

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        symptoms = row["symptoms"]
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

def train(epochs, batch_size, lr):
    label_list = load_label_list()
    if not label_list:
        print("No data found. Run --mode prepare-data or --mode add-data first.")
        return
    dataset = FusionDataset(CSV_PATH, label_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = DiagnosisFusionModel(num_conditions=len(label_list))
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, symptoms, labels in loader:
            logits = model(images, symptoms)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}") 
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({"model_state": model.classifier.state_dict(), "label_list": label_list}, CHECKPOINT_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test the medical image+symptom fusion model")
    parser.add_argument("--mode", required=True, choices=["prepare-data", "add-data", "train", "test"])
    parser.add_argument("--image", help="Path to an image file (for --mode add-data)")
    parser.add_argument("--symptoms", help="Symptom description text (for --mode add-data)")
    parser.add_argument("--diagnosis", help="Diagnosis label (for --mode add-data)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.mode == "prepare-data":
        prepare_data()
    elif args.mode == "add-data":
        if not (args.image and args.symptoms and args.diagnosis):
            print("--mode add-data requires --image, --symptoms, and --diagnosis")
        else:
            add_data(args.image, args.symptoms, args.diagnosis)
    elif args.mode == "train":
        train(args.epochs, args.batch_size, args.lr)
    elif args.mode == "test":
        test(args.batch_size)