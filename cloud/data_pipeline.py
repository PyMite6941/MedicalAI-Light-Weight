import csv
import io
import json
import os
import random
from collections import Counter, OrderedDict

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
LOCAL_DATA_DIR = "/data" if os.path.exists("/data") else "./data"
LOCAL_IMAGES_DIR = os.path.join(LOCAL_DATA_DIR, "images")
CSV_PATH = os.path.join(LOCAL_DATA_DIR, "dataset.csv")

CANONICAL_LABELS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
    "Calcified_Granuloma", "Granulomatous_Disease", "Bronchiectasis",
    "Hiatal_Hernia", "Pneumoperitoneum", "Pericardial_Effusion",
    "Rib_Fracture", "Clavicle_Fracture", "Sternal_Fracture",
    "COPD", "Interstitial_Lung_Disease", "Sarcoidosis",
    "Lung_Cancer", "Metastatic_Disease", "Mediastinal_Mass",
    "Thoracic_Aortic_Aneurysm", "Aortic_Dissection",
    "Pulmonary_Embolism", "Pulmonary_Hypertension",
    "Tuberculosis", "Aspergillosis", "Fungal_Infection",
    "ARDS", "Pulmonary_Edema", "Alveolar_Hemorrhage",
    "Organizing_Pneumonia", "Hypersensitivity_Pneumonitis",
    "Lymphangitic_Carcinomatosis", "Lymphadenopathy",
    "Pleural_Plaque", "Pneumoconiosis", "Silicosis", "Asbestosis",
    "Swyer_James", "LAM", "LCH", "Alveolar_Proteinosis",
    "Scimitar_Syndrome", "Tetralogy_of_Fallot", "Coarctation",
    "Transposition_of_Great_Arteries", "Hiatal_Hernia",
    "Pneumopericardium", "Subcutaneous_Emphysema",
    "Foreign_Body", "Post_Surgical_Changes",
    "Support_Devices", "Monitoring_Devices",
    "Pancoast_Tumor", "Superior_Vena_Cava_Syndrome",
    "Thoracic_Outlet_Syndrome", "Diaphragmatic_Hernia",
    "Eventration_of_Diaphragm", "Diaphragmatic_Paralysis",
    "Empyema", "Hemothorax", "Chylothorax",
    "Fibrothorax", "Trapped_Lung", "Round_Atelectasis",
    "Mucus_Plugging", "Tree_in_Bud", "Air_Trapping",
    "Broncholithiasis", "Bronchial_Stenosis", "Tracheal_Stenosis",
    "Tracheomegaly", "Tracheoesophageal_Fistula",
    "Esophageal_Dilatation", "Achalasia", "Zenker_Diverticulum",
    "Mediastinal_Lymphadenopathy", "Mediastinal_Hematoma",
    "Mediastinal_Abscess", "Mediastinal_Fibrosis",
    "Thoracic_Spine_Fracture", "Osteoporosis", "Kyphoscoliosis",
    "Ankylosing_Spondylitis", "Diffuse_Idiopathic_Skeletal_Hyperostosis",
]

LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}
NUM_CLASSES = len(CANONICAL_LABELS)

SYMPTOM_TEMPLATES = [
    "What abnormality is present in this chest X-ray?",
    "Patient presents with shortness of breath and cough. What is the diagnosis?",
    "Routine pre-operative chest X-ray. Any abnormal findings?",
    "Patient with history of smoking. Evaluate for lung pathology.",
    "Fever and productive cough for {days} days. Assess for pneumonia.",
    "Chest pain and dyspnea on exertion. Cardiac or pulmonary cause?",
    "Post-surgical follow-up. Evaluate lung expansion and complications.",
    "Patient with known COPD. Assess for acute changes or infection.",
    "Trauma patient after MVA. Evaluate for pneumothorax, hemothorax, or fractures.",
    "Immunocompromised patient with fever. Opportunistic infection?",
    "Patient with weight loss and night sweats. Evaluate for TB or malignancy.",
    "Dysphagia and regurgitation. Evaluate for hiatal hernia or mediastinal mass.",
    "Pre-employment screening chest X-ray.",
    "CHF follow-up. Evaluate for pulmonary edema.",
    "Patient with known ILD. Assess for progression.",
    "Hemoptysis for 2 weeks. Evaluate for bronchiectasis or mass.",
    "Contact TB patient. Screening chest X-ray.",
    "HIV positive patient with cough and fever.",
    "Post-chemotherapy evaluation. Neutropenic fever.",
    "Patient with asbestos exposure history. Routine surveillance.",
    "Chest trauma after MVA. Evaluate for aortic injury.",
    "Suspected foreign body aspiration.",
    "Rule out TB in patient with positive PPD.",
    "Dyspnea and orthopnea. Evaluate for heart failure.",
    "Liver cirrhosis patient with dyspnea. Hepatic hydrothorax?",
    "ARDS follow-up chest X-ray.",
    "Ventilator-associated pneumonia surveillance.",
    "Suspected pulmonary embolism. Evaluate for Hampton's hump.",
    "Chronic cough with mucus production. Bronchiectasis evaluation.",
]


def load_gcs_json(bucket, blob_path):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(blob_path)
    return json.loads(blob.download_as_string())


def download_gcs_data(bucket, prefix="data/"):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket)
    os.makedirs(LOCAL_IMAGES_DIR, exist_ok=True)
    blobs = list(bucket.list_blobs(prefix=prefix))
    downloaded = 0
    for blob in blobs:
        if blob.name.endswith(".csv"):
            blob.download_to_filename(CSV_PATH)
        elif blob.name.endswith((".jpg", ".jpeg", ".png")):
            local_name = os.path.join(LOCAL_IMAGES_DIR, os.path.basename(blob.name))
            if not os.path.exists(local_name):
                blob.download_to_filename(local_name)
                downloaded += 1
    return downloaded


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
    back to substring matching, preferring the *longest* (most specific)
    key — otherwise a broad label like "Mass" would shadow a more specific
    one like "Mediastinal_Mass" just because it sorts earlier, and a naive
    single-word match (the old behavior) could match almost anything.
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


def read_dataset_csv(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row.get("image_path", "").strip()
            if p and os.path.exists(p):
                rows.append(row)
    return rows


class MegaFusionDataset(Dataset):
    def __init__(self, csv_path, image_dir, max_samples=None, normalize_labels=True):
        self.image_dir = image_dir
        self.normalize_labels = normalize_labels
        self.rows = read_dataset_csv(csv_path)
        if max_samples and len(self.rows) > max_samples:
            self.rows = random.sample(self.rows, max_samples)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = row["image_path"].strip()
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            dummy = Image.new("RGB", (224, 224), color=128)
            return dummy, "", torch.zeros(NUM_CLASSES, dtype=torch.float32)

        symptoms = row.get("symptoms", "").strip()
        if not symptoms:
            symptoms = random.choice(SYMPTOM_TEMPLATES)
            symptoms = symptoms.format(days=random.choice([3, 5, 7, 10, 14]))

        raw_labels = row.get("labels", "").strip()
        label_names = parse_labels(raw_labels)
        if not label_names:
            diag = row.get("diagnosis", "").strip()
            if diag:
                label_names = [diag]
        multihot = labels_to_multihot(label_names) if self.normalize_labels else torch.zeros(NUM_CLASSES)

        return image, symptoms, multihot


def custom_collate(batch):
    images = [item[0] for item in batch]
    symptoms = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    return images, symptoms, labels


def create_dataloaders(csv_path, image_dir, batch_size, val_split=0.1, max_samples=None, distributed=False):
    dataset = MegaFusionDataset(csv_path, image_dir, max_samples=max_samples)
    val_size = max(int(val_split * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=custom_collate,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, collate_fn=custom_collate,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader, len(CANONICAL_LABELS), CANONICAL_LABELS


def get_label_stats(csv_path):
    rows = read_dataset_csv(csv_path)
    counter = Counter()
    total = 0
    for row in rows:
        label_str = row.get("labels", "").strip()
        names = parse_labels(label_str)
        if not names:
            diag = row.get("diagnosis", "").strip()
            if diag:
                names = [diag]
        for n in names:
            idx = map_to_canonical(n)
            if idx >= 0:
                counter[CANONICAL_LABELS[idx]] += 1
        total += 1
    return counter, total