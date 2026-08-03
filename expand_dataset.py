import csv
import os
import random

CSV_PATH = "./data/dataset.csv"
IMAGES_DIR = "./data/images"
os.makedirs(IMAGES_DIR, exist_ok=True)

CSV_COLUMNS = ["image_path", "source", "symptoms", "diagnosis", "labels"]

SYMPTOM_TEMPLATES = [
    "What abnormality is present in this chest X-ray?",
    "Patient presents with shortness of breath and cough.",
    "Routine pre-operative chest X-ray.",
    "Patient with history of smoking. Evaluate for lung pathology.",
    "Fever and productive cough for {days} days.",
    "Chest pain and dyspnea on exertion.",
    "Patient with known COPD. Assess for acute changes.",
    "Trauma patient after MVA. Evaluate for pneumothorax.",
    "Immunocompromised patient with fever.",
    "Patient with weight loss and night sweats.",
    "Hemoptysis for 2 weeks.",
    "Contact TB patient. Screening chest X-ray.",
    "HIV positive patient with cough and fever.",
    "Post-chemotherapy evaluation.",
    "Dyspnea and orthopnea.",
    "Chronic cough with mucus production.",
    "Rheumatoid arthritis patient with new dyspnea.",
    "Asbestos exposure history. Surveillance.",
    "ARDS follow-up chest X-ray.",
    "Suspected pulmonary embolism.",
]


def count_existing():
    if not os.path.exists(CSV_PATH):
        return 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def write_header():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


def append(image_path, source, symptoms, diagnosis, labels_str):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([image_path, source, symptoms, diagnosis, labels_str])


# ═══════════════════════════════════════════════════════════════
# DATASET 1: CheXpert (Stanford) — 224,316 chest radiographs
# ═══════════════════════════════════════════════════════════════
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
CHEXPERT_MAP = {
    "No Finding": "No Finding", "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema", "Consolidation": "Consolidation",
    "Atelectasis": "Atelectasis", "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion", "Fracture": "Rib_Fracture",
    "Lung Opacity": "Infiltration", "Lung Lesion": "Nodule",
    "Pneumonia": "Pneumonia", "Support Devices": "Support_Devices",
}


def load_chexpert(max_images=50000):
    print(f"\n{'='*60}")
    print(f"LOADING: CheXpert (Stanford) - up to {max_images} images")
    print(f"{'='*60}")
    try:
        from datasets import load_dataset
        ds = load_dataset("StanfordAIMI/chexpert", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  CheXpert download failed: {e}")
        return 0

    written = 0
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        if image is None:
            continue
        found = []
        for cl in CHEXPERT_LABELS:
            val = example.get(cl, 0)
            if val == 1:
                mapped = CHEXPERT_MAP.get(cl)
                if mapped:
                    found.append(mapped)
        if not found:
            continue
        labels_str = "|".join(found)
        img_path = os.path.join(IMAGES_DIR, f"chexpert_{i}.jpg")
        image.convert("RGB").save(img_path, quality=90)
        append(img_path, "chexpert", "", found[0], labels_str)
        written += 1
        if written % 10000 == 0:
            print(f"  CheXpert: {written}/{max_images}...")
    print(f"  CheXpert loaded: {written} rows")
    return written


# ═══════════════════════════════════════════════════════════════
# DATASET 2: PadChest (Universidad de Alicante) — 160,000+ images
# ═══════════════════════════════════════════════════════════════
PADCHEST_LABEL_MAP = {
    "normal": "No Finding", "cardiomegaly": "Cardiomegaly",
    "pleural effusion": "Effusion", "atelectasis": "Atelectasis",
    "pneumothorax": "Pneumothorax", "consolidation": "Consolidation",
    "pulmonary edema": "Edema", "emphysema": "Emphysema",
    "fibrosis": "Fibrosis", "nodule": "Nodule", "mass": "Mass",
    "pneumonia": "Pneumonia", "infiltrate": "Infiltration",
    "hernia": "Hernia", "bronchiectasis": "Bronchiectasis",
    "pleural thickening": "Pleural_Thickening",
    "calcified granuloma": "Calcified_Granuloma",
    "granuloma": "Granulomatous_Disease",
    "costophrenic angle blunting": "Effusion",
    "interstitial pattern": "Interstitial_Lung_Disease",
    "reticular pattern": "Fibrosis", "reticulonodular pattern": "Fibrosis",
    "ground glass pattern": "Infiltration",
    "alveolar pattern": "Consolidation",
    "COPD signs": "COPD", "hyperinflation": "COPD",
    "silicosis": "Silicosis", "asbestosis": "Asbestosis",
    "pleural plaque": "Pleural_Plaque",
    "tuberculosis": "Tuberculosis",
    "aortic elongation": "Aortic_Aneurysm",
    "rib fracture": "Rib_Fracture",
    "lymphadenopathy": "Lymphadenopathy",
    "surgery": "Post_Surgical_Changes",
    "pacemaker": "Support_Devices",
    "port-a-cath": "Support_Devices",
    "kyphosis": "Kyphoscoliosis",
    "scoliosis": "Kyphoscoliosis",
}


def load_padchest(max_images=40000):
    print(f"\n{'='*60}")
    print(f"LOADING: PadChest (Universidad de Alicante) - up to {max_images} images")
    print(f"{'='*60}")
    try:
        from datasets import load_dataset
        ds = load_dataset("gmongas/PadChest", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  PadChest download failed: {e}")
        return 0

    written = 0
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        labels_raw = example.get("labels", "")
        if image is None or not labels_raw:
            continue
        found = []
        for lbl in labels_raw.lower().split(","):
            lbl = lbl.strip().replace("-", " ")
            for key, mapped in PADCHEST_LABEL_MAP.items():
                if key in lbl or lbl in key:
                    if mapped not in found:
                        found.append(mapped)
        if not found:
            found = ["No Finding"]
        labels_str = "|".join(found)
        img_path = os.path.join(IMAGES_DIR, f"padchest_{i}.jpg")
        image.convert("RGB").save(img_path, quality=90)
        append(img_path, "padchest", "", found[0], labels_str)
        written += 1
        if written % 10000 == 0:
            print(f"  PadChest: {written}/{max_images}...")
    print(f"  PadChest loaded: {written} rows")
    return written


# ═══════════════════════════════════════════════════════════════
# DATASET 3: VinDr-CXR (Vietnam) — 18,000+ with 22 local labels
# ═══════════════════════════════════════════════════════════════
VINDR_LABEL_MAP = {
    "0": "No Finding", "1": "Aortic_Aneurysm", "2": "Atelectasis",
    "3": "Calcified_Granuloma", "4": "Cardiomegaly",
    "5": "Cavitary_Lesion", "6": "Consolidation",
    "7": "COPD", "8": "Edema", "9": "Effusion",
    "10": "Emphysema", "11": "Fibrosis",
    "12": "Hemothorax", "13": "Infiltration",
    "14": "Interstitial_Lung_Disease", "15": "Lymphadenopathy",
    "16": "Mass", "17": "Mediastinal_Mass",
    "18": "Nodule", "19": "Pleural_Thickening",
    "20": "Pneumonia", "21": "Pneumothorax",
    "22": "Rib_Fracture", "23": "Tuberculosis",
}


def load_vindr(max_images=18000):
    print(f"\n{'='*60}")
    print(f"LOADING: VinDr-CXR (Vietnam) - up to {max_images} images")
    print(f"{'='*60}")
    try:
        from datasets import load_dataset
        ds = load_dataset("PhysioNet/vindr-cxr", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  VinDr-CXR download failed: {e}")
        print(f"  Trying alternative path...")
        try:
            ds = load_dataset("vindr/vindr-cxr", split="train", streaming=True, trust_remote_code=True)
        except Exception as e2:
            print(f"  VinDr-CXR alt also failed: {e2}")
            return 0

    written = 0
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        if image is None:
            continue
        labels_raw = example.get("class_ids", example.get("labels", []))
        if not labels_raw:
            found = ["No Finding"]
        else:
            found = []
            for lid in labels_raw:
                mapped = VINDR_LABEL_MAP.get(str(lid))
                if mapped and mapped not in found:
                    found.append(mapped)
            if not found:
                found = ["No Finding"]
        labels_str = "|".join(found)
        img_path = os.path.join(IMAGES_DIR, f"vindr_{i}.jpg")
        image.convert("RGB").save(img_path, quality=90)
        append(img_path, "vindr", "", found[0], labels_str)
        written += 1
        if written % 5000 == 0:
            print(f"  VinDr-CXR: {written}/{max_images}...")
    print(f"  VinDr-CXR loaded: {written} rows")
    return written


# ═══════════════════════════════════════════════════════════════
# DATASET 4: CheXphoto (Stanford) — 10,000 phone photos of X-rays
# ═══════════════════════════════════════════════════════════════
def load_chexphoto(max_images=10000):
    print(f"\n{'='*60}")
    print(f"LOADING: CheXphoto (Stanford phone photos) - up to {max_images}")
    print(f"{'='*60}")
    try:
        from datasets import load_dataset
        ds = load_dataset("StanfordAIMI/chexphoto", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  CheXphoto download failed: {e}")
        return 0

    written = 0
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        if image is None:
            continue
        labels_raw = example.get("labels", {})
        found = []
        for lbl, val in labels_raw.items():
            if val == 1:
                mapped = CHEXPERT_MAP.get(lbl)
                if mapped and mapped not in found:
                    found.append(mapped)
        if not found:
            found = ["No Finding"]
        labels_str = "|".join(found)
        img_path = os.path.join(IMAGES_DIR, f"chexphoto_{i}.jpg")
        image.convert("RGB").save(img_path, quality=90)
        append(img_path, "chexphoto", "", found[0], labels_str)
        written += 1
        if written % 2500 == 0:
            print(f"  CheXphoto: {written}/{max_images}...")
    print(f"  CheXphoto loaded: {written} rows")
    return written


# ═══════════════════════════════════════════════════════════════
# DATASET 5: SIIM-ACR Pneumothorax — ~12,000 images
# ═══════════════════════════════════════════════════════════════
def load_siim_pneumothorax(max_images=12000):
    print(f"\n{'='*60}")
    print(f"LOADING: SIIM-ACR Pneumothorax - up to {max_images}")
    print(f"{'='*60}")
    try:
        from datasets import load_dataset
        ds = load_dataset("siim-acr/pneumothorax-segmentation", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  SIIM-ACR download failed: {e}")
        return 0

    written = 0
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        if image is None:
            continue
        has_pneumo = example.get("pneumothorax", 0) == 1
        labels_str = "Pneumothorax" if has_pneumo else "No Finding"
        img_path = os.path.join(IMAGES_DIR, f"siim_{i}.jpg")
        image.convert("RGB").save(img_path, quality=90)
        append(img_path, "siim_pneumothorax", "", labels_str, labels_str)
        written += 1
        if written % 2500 == 0:
            print(f"  SIIM-ACR: {written}/{max_images}...")
    print(f"  SIIM-ACR loaded: {written} rows")
    return written


# ═══════════════════════════════════════════════════════════════
# DATASET 6: RSNA Pneumonia Detection — ~30,000 images
# ═══════════════════════════════════════════════════════════════
def load_rsna_pneumonia(max_images=30000):
    print(f"\n{'='*60}")
    print(f"LOADING: RSNA Pneumonia Detection - up to {max_images}")
    print(f"{'='*60}")
    try:
        from datasets import load_dataset
        ds = load_dataset("rsna/pneumonia-detection", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  RSNA download failed: {e}")
        return 0

    written = 0
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        if image is None:
            continue
        target = example.get("Target", 0)
        labels_str = "Pneumonia" if target == 1 else "No Finding"
        img_path = os.path.join(IMAGES_DIR, f"rsna_{i}.jpg")
        image.convert("RGB").save(img_path, quality=90)
        append(img_path, "rsna_pneumonia", "", labels_str, labels_str)
        written += 1
        if written % 5000 == 0:
            print(f"  RSNA: {written}/{max_images}...")
    print(f"  RSNA loaded: {written} rows")
    return written


# ═══════════════════════════════════════════════════════════════
# DATASET 7: NIH ChestX-ray14 FULL — 112,120 images
# ═══════════════════════════════════════════════════════════════
NIH_LABEL_NAMES = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def load_nih_full(max_images=80000):
    print(f"\n{'='*60}")
    print(f"LOADING: NIH ChestX-ray14 FULL - up to {max_images} images")
    print(f"{'='*60}")
    try:
        from datasets import load_dataset
        ds = load_dataset("g-ronimo/NIH-Chest-X-ray-dataset_resized300px", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  NIH download failed: {e}")
        print("  Trying full NIH path...")
        try:
            ds = load_dataset("nih-chest-xray/nih-chest-xray-dataset", split="train", streaming=True, trust_remote_code=True)
        except Exception as e2:
            print(f"  NIH alt also failed: {e2}")
            return 0

    written = 0
    for i, example in enumerate(ds):
        if written >= max_images:
            break
        image = example.get("image")
        label_indices = example.get("labels", [])
        if image is None or not label_indices:
            continue
        found = []
        for idx in label_indices:
            if idx < len(NIH_LABEL_NAMES):
                name = NIH_LABEL_NAMES[idx]
                if name and name != "No Finding" and name not in found:
                    found.append(name)
        if not found:
            found = ["No Finding"]
        labels_str = "|".join(found)
        img_path = os.path.join(IMAGES_DIR, f"nih_full_{i}.jpg")
        image.convert("RGB").save(img_path, quality=90)
        append(img_path, "nih_full", "", found[0], labels_str)
        written += 1
        if written % 10000 == 0:
            print(f"  NIH Full: {written}/{max_images}...")
    print(f"  NIH Full loaded: {written} rows")
    return written


# ═══════════════════════════════════════════════════════════════
# AUGMENTED: 160 rare/obscure diagnosis entries (from original expand_dataset.py)
# ═══════════════════════════════════════════════════════════════
AUGMENTED_ENTRIES = [
    {
        "diagnosis": "FINDINGS: The cardiomediastinal silhouette is within normal limits. The lungs are clear without focal consolidation, pleural effusion, or pneumothorax. The bony thorax is intact. IMPRESSION: Normal chest radiograph.",
        "labels": "No Finding",
    },
    {
        "diagnosis": "FINDINGS: The cardiac silhouette is moderately enlarged. Pulmonary vascularity is increased with cephalization of the upper lobe vessels. Small bilateral pleural effusions. IMPRESSION: Cardiomegaly with signs of CHF.",
        "labels": "Cardiomegaly|Edema|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Severe cardiomegaly. Diffuse bilateral interstitial opacities with Kerley B lines. Bilateral pleural effusions, right greater than left. IMPRESSION: Severe cardiomegaly with interstitial pulmonary edema consistent with CHF.",
        "labels": "Cardiomegaly|Edema|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Dense airspace consolidation in the right lower lobe with obscuration of the right hemidiaphragm. Air bronchograms present. IMPRESSION: Right lower lobe pneumonia.",
        "labels": "Consolidation|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Patchy airspace opacities in the left upper lobe with ill-defined margins. No cavitation. No effusion. IMPRESSION: Left upper lobe pneumonia.",
        "labels": "Consolidation|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Multifocal bilateral patchy and confluent airspace opacities in a peribronchovascular distribution. Small bilateral effusions. IMPRESSION: Multifocal pneumonia, consider atypical etiology.",
        "labels": "Infiltration|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Linear opacity in the right lower lobe extending to the pleura. Minor fissure elevation. IMPRESSION: Platelike atelectasis in the right lower lobe.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Moderate right pleural effusion with blunting of the right costophrenic angle and meniscus sign. Underlying compressive atelectasis. IMPRESSION: Moderate right pleural effusion with adjacent atelectasis.",
        "labels": "Effusion|Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Right apical pneumothorax with visceral pleural line visible approximately 2 cm from the chest wall. No mediastinal shift. IMPRESSION: Small right apical pneumothorax.",
        "labels": "Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Hyperinflated lungs with flattened hemidiaphragms. Increased AP chest diameter. Bullous changes at the apices. IMPRESSION: COPD with emphysematous changes.",
        "labels": "Emphysema|COPD",
    },
    {
        "diagnosis": "FINDINGS: Severe hyperinflation with flattened and depressed hemidiaphragms. Widened intercostal spaces. Increased retrosternal airspace. IMPRESSION: Severe emphysema.",
        "labels": "Emphysema|COPD",
    },
    {
        "diagnosis": "FINDINGS: Solitary pulmonary nodule in the right upper lobe measuring approximately 1.5 cm. Margins are smooth. IMPRESSION: Solitary pulmonary nodule. CT recommended.",
        "labels": "Nodule",
    },
    {
        "diagnosis": "FINDINGS: Spiculated mass in the left upper lobe measuring 3.2 x 2.8 cm. Associated pleural tail. No cavitation. IMPRESSION: Spiculated left upper lobe mass suspicious for primary lung malignancy.",
        "labels": "Mass|Nodule",
    },
    {
        "diagnosis": "FINDINGS: Multiple bilateral pulmonary nodules of varying sizes, ranging from 0.5 to 2.0 cm. Random distribution. Small right pleural effusion. IMPRESSION: Numerous bilateral pulmonary nodules consistent with metastatic disease.",
        "labels": "Nodule|Mass|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Cavitating mass in the right upper lobe with thick, irregular walls and air-fluid level. IMPRESSION: Cavitating lung mass, differential includes carcinoma, abscess, or fungal infection.",
        "labels": "Mass|Cavitary_Lesion",
    },
    {
        "diagnosis": "FINDINGS: Pancoast tumor in the right apex with associated apical cap thickening and destruction of the posterior right first and second ribs. IMPRESSION: Right apical mass (Pancoast tumor) with chest wall invasion.",
        "labels": "Mass|Pancoast_Tumor",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral reticular opacities with honeycombing in the lung bases. Traction bronchiectasis present. IMPRESSION: UIP pattern consistent with IPF.",
        "labels": "Fibrosis|Interstitial_Lung_Disease",
    },
    {
        "diagnosis": "FINDINGS: Bilateral ground-glass opacities in the lower lobes with reticular superimposed opacities. Minimal honeycombing. IMPRESSION: NSIP pattern.",
        "labels": "Fibrosis|Infiltration|Interstitial_Lung_Disease",
    },
    {
        "diagnosis": "FINDINGS: Extensive bilateral interstitial opacities with perilymphatic distribution. Septal thickening and subpleural nodules. Bilateral hilar lymphadenopathy. IMPRESSION: Pulmonary sarcoidosis (stage II).",
        "labels": "Fibrosis|Nodule|Lymphadenopathy|Sarcoidosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral perihilar airspace opacities with butterfly-wing pattern. Upper lobe redistribution. Kerley B lines. Cardiomegaly. Small bilateral effusions. IMPRESSION: Acute pulmonary edema due to CHF.",
        "labels": "Edema|Cardiomegaly|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral airspace opacities with central predominance. Normal heart size. No pleural effusion. IMPRESSION: Noncardiogenic edema / ARDS pattern.",
        "labels": "Edema|Infiltration|ARDS",
    },
    {
        "diagnosis": "FINDINGS: Right apical fibronodular opacities with volume loss and upward hilar retraction. Calcified granuloma. IMPRESSION: Prior granulomatous disease, likely old TB.",
        "labels": "Fibrosis|Nodule|Tuberculosis|Calcified_Granuloma",
    },
    {
        "diagnosis": "FINDINGS: Cavitary lesion in the right upper lobe with surrounding airspace disease. Tree-in-bud opacities. Right paratracheal adenopathy. IMPRESSION: Active pulmonary TB suspected.",
        "labels": "Infiltration|Cavitary_Lesion|Tree_in_Bud|Tuberculosis",
    },
    {
        "diagnosis": "FINDINGS: Military pattern with innumerable tiny 1-2 mm nodules diffusely throughout both lungs. IMPRESSION: Military nodules, highly suspicious for miliary TB.",
        "labels": "Nodule|Tuberculosis",
    },
    {
        "diagnosis": "FINDINGS: Dilated, thickened bronchi seen as tram-track opacities in bilateral lower lobes. Mucus plugging. IMPRESSION: Bilateral lower lobe bronchiectasis.",
        "labels": "Bronchiectasis|Mucus_Plugging",
    },
    {
        "diagnosis": "FINDINGS: Large retrocardiac air-fluid level behind the heart. Herniated stomach contents above diaphragm. IMPRESSION: Large hiatal hernia.",
        "labels": "Hernia|Hiatal_Hernia",
    },
    {
        "diagnosis": "FINDINGS: Free air under the right hemidiaphragm on upright view. IMPRESSION: Pneumoperitoneum suggesting hollow viscus perforation.",
        "labels": "Pneumoperitoneum",
    },
    {
        "diagnosis": "FINDINGS: Globular enlargement of the cardiac silhouette with water-bottle configuration. Clear sharp borders. Lungs clear. IMPRESSION: Large pericardial effusion.",
        "labels": "Pericardial_Effusion",
    },
    {
        "diagnosis": "FINDINGS: Hampton's hump: wedge-shaped pleural-based opacity in the right costophrenic angle. Small right pleural effusion. IMPRESSION: Pulmonary infarct, likely PE.",
        "labels": "Effusion|Infiltration|Pulmonary_Embolism",
    },
    {
        "diagnosis": "FINDINGS: Westermark sign: oligemia of left lung with decreased vascular markings. Prominent central pulmonary artery. IMPRESSION: Westermark sign suggesting pulmonary embolism.",
        "labels": "Pulmonary_Embolism",
    },
    {
        "diagnosis": "FINDINGS: Enlarged right descending pulmonary artery (Palla sign). Right lower lobe opacity. Small right pleural effusion. IMPRESSION: Concerning for pulmonary embolism.",
        "labels": "Effusion|Infiltration|Pulmonary_Embolism",
    },
    {
        "diagnosis": "FINDINGS: Thymic sail sign with triangular soft tissue from right superior mediastinum. Lungs clear. IMPRESSION: Normal thymus in pediatric chest.",
        "labels": "No Finding",
    },
    {
        "diagnosis": "FINDINGS: Scimitar sign: curved tubular opacity in right lower lobe toward cardiophrenic angle. Hypoplastic right lung. Mediastinal shift right. IMPRESSION: Scimitar syndrome.",
        "labels": "Scimitar_Syndrome",
    },
    {
        "diagnosis": "FINDINGS: Boot-shaped heart with upturned cardiac apex, prominent right ventricle, concave pulmonary artery segment. IMPRESSION: Tetralogy of Fallot.",
        "labels": "Tetralogy_of_F allot|Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Small rounded nodular opacities in upper and mid lung zones bilaterally. Eggshell calcifications of hilar nodes. IMPRESSION: Simple silicosis.",
        "labels": "Nodule|Silicosis|Pneumoconiosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral upper lobe large opacities (PMF) with surrounding emphysema. Hilar retraction. Eggshell calcification. IMPRESSION: Complicated silicosis with PMF.",
        "labels": "Fibrosis|Mass|Silicosis|Pneumoconiosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral diaphragmatic and lateral pleural plaques with calcification. No effusion. IMPRESSION: Asbestos-related pleural plaques.",
        "labels": "Pleural_Plaque|Asbestosis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Bilateral lower lobe interstitial fibrosis with honeycombing. Calcified pleural plaques. IMPRESSION: Asbestosis with parenchymal fibrosis.",
        "labels": "Fibrosis|Pleural_Plaque|Asbestosis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Extensive bilateral ground-glass opacities with interlobular septal thickening (crazy-paving). IMPRESSION: Crazy-paving, consider alveolar proteinosis.",
        "labels": "Infiltration|Alveolar_Proteinosis|Interstitial_Lung_Disease",
    },
    {
        "diagnosis": "FINDINGS: Bilateral diffuse micronodular opacities mid-to-upper lung. Multiple thin-walled cysts. IMPRESSION: LCH in a smoker.",
        "labels": "Nodule|Cystic_Lung_Disease|LCH",
    },
    {
        "diagnosis": "FINDINGS: Bilateral large thin-walled cysts with normal intervening parenchyma. Lower lobe distribution. IMPRESSION: LAM.",
        "labels": "Cystic_Lung_Disease|LAM",
    },
    {
        "diagnosis": "FINDINGS: Swyer-James syndrome: hyperlucent left lung with diminished vascular markings. Air trapping on expiration. IMPRESSION: Swyer-James/MacLeod syndrome.",
        "labels": "Emphysema|Swyer_James",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral ground-glass opacities with peripheral predominance. Reverse halo sign visible. IMPRESSION: Organizing pneumonia.",
        "labels": "Consolidation|Infiltration|Organizing_Pneumonia",
    },
    {
        "diagnosis": "FINDINGS: Multiple cavitary nodules with thick walls. Feeding vessel signs. IMPRESSION: Septic pulmonary emboli.",
        "labels": "Nodule|Cavitary_Lesion|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Bilateral symmetric lower lobe consolidation with air bronchograms. Kerley B lines. Small bilateral effusions. Normal heart size. IMPRESSION: AIP/Hamman-Rich syndrome.",
        "labels": "Consolidation|Effusion|Infiltration|ARDS",
    },
    {
        "diagnosis": "FINDINGS: Multiple subcentimeter nodules with perilymphatic distribution and bilateral hilar adenopathy. IMPRESSION: Stage I sarcoidosis with BHL (1-2-3 sign).",
        "labels": "Nodule|Lymphadenopathy|Sarcoidosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral hilar and right paratracheal lymphadenopathy with upper lobe reticulonodular opacities. IMPRESSION: Stage II sarcoidosis.",
        "labels": "Nodule|Fibrosis|Lymphadenopathy|Sarcoidosis",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral fine nodules with upper lobe predominance. Mild hilar adenopathy. IMPRESSION: Hypersensitivity pneumonitis.",
        "labels": "Nodule|Hypersensitivity_Pneumonitis",
    },
    {
        "diagnosis": "FINDINGS: Extensive bilateral consolidations and ground-glass opacities with air bronchograms. IMPRESSION: Diffuse alveolar hemorrhage.",
        "labels": "Consolidation|Alveolar_Hemorrhage",
    },
    {
        "diagnosis": "FINDINGS: Unilateral right perihilar mass with obstructive pneumonitis and Golden S sign. Right upper lobe volume loss. IMPRESSION: Central lung mass with obstructive atelectasis.",
        "labels": "Mass|Atelectasis|Lung_Cancer",
    },
    {
        "diagnosis": "FINDINGS: Air crescent sign in a preexisting cavity with round opacity inside. Right upper lobe. IMPRESSION: Aspergilloma with air crescent sign.",
        "labels": "Nodule|Mass|Aspergillosis|Cavitary_Lesion",
    },
    {
        "diagnosis": "FINDINGS: Multiple randomly distributed nodules with halo sign (ground-glass surrounding nodule). IMPRESSION: Fungal infection in immunocompromised patient.",
        "labels": "Nodule|Infiltration|Fungal_Infection",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral fine miliary nodules with random distribution. Bilateral hilar adenopathy. IMPRESSION: Miliary TB versus metastatic disease.",
        "labels": "Nodule|Tuberculosis|Lymphadenopathy",
    },
    {
        "diagnosis": "FINDINGS: Bilateral pleural effusions with associated basal atelectasis. Cardiomegaly. Pulmonary vascular congestion. IMPRESSION: CHF with bilateral effusions.",
        "labels": "Effusion|Cardiomegaly|Atelectasis|Edema",
    },
    {
        "diagnosis": "FINDINGS: Wide mediastinum >8 cm on AP view. Loss of aortic knob contour. Left apical cap. Left pleural effusion. IMPRESSION: Traumatic aortic injury.",
        "labels": "Effusion|Aortic_Dissection",
    },
    {
        "diagnosis": "FINDINGS: Large left pleural effusion causing near-complete opacification of the left hemithorax with mediastinal shift right. IMPRESSION: Massive left pleural effusion. Thoracentesis indicated.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Loculated right pleural effusion with biconvex opacity along lateral chest wall. IMPRESSION: Loculated effusion, consider empyema.",
        "labels": "Effusion|Empyema",
    },
    {
        "diagnosis": "FINDINGS: Tension pneumothorax on the right with deep sulcus sign, mediastinal shift left, flattened right heart border. IMPRESSION: Large tension pneumothorax, STAT decompression.",
        "labels": "Pneumothorax",
    },
    {
        "diagnosis": "FINDINGS: Hydropneumothorax on the left with air-fluid level. Partially collapsed left lung. Mediastinal shift right. IMPRESSION: Hydropneumothorax, consider bronchopleural fistula.",
        "labels": "Pneumothorax|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Severe bilateral bullous emphysema. Large thin-walled bullae >1/3 of both hemithoraces. IMPRESSION: Severe bilateral bullous emphysema.",
        "labels": "Emphysema|COPD",
    },
    {
        "diagnosis": "FINDINGS: Subcutaneous emphysema with air tracking in chest wall soft tissues. No pneumothorax. IMPRESSION: Subcutaneous emphysema.",
        "labels": "Subcutaneous_Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Left lower lobe consolidation obscuring left hemidiaphragm and descending aorta. Air bronchograms. Small left effusion. IMPRESSION: Left lower lobe pneumonia with parapneumonic effusion.",
        "labels": "Consolidation|Effusion|Infiltration|Pneumonia",
    },
    {
        "diagnosis": "FINDINGS: Bilateral perihilar interstitial and airspace opacities central distribution. No effusion. IMPRESSION: Interstitial pneumonia, favor viral etiology.",
        "labels": "Infiltration|Consolidation|Pneumonia",
    },
    {
        "diagnosis": "FINDINGS: Dense consolidation in right upper lobe with air bronchograms and volume loss. Right tracheal shift. IMPRESSION: RUL pneumonia with volume loss.",
        "labels": "Consolidation|Atelectasis|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Round pneumonia as spherical opacity in left lower lobe. Surrounding ground-glass. IMPRESSION: Round pneumonia left lower lobe.",
        "labels": "Consolidation|Pneumonia",
    },
    {
        "diagnosis": "FINDINGS: Cavitary lesion in right upper lobe with thick irregular wall and air-fluid level. Surrounding consolidation. IMPRESSION: Cavitary pneumonia, consider TB or necrotizing infection.",
        "labels": "Infiltration|Consolidation|Cavitary_Lesion",
    },
    {
        "diagnosis": "FINDINGS: Bilateral lower lobe consolidations with air bronchograms. Small bilateral effusions. Cardiomegaly. IMPRESSION: Bilateral pneumonias with cardiomegaly.",
        "labels": "Consolidation|Infiltration|Cardiomegaly|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Golden S sign in right upper lobe with S-shaped minor fissure. Central hilar mass suspected. IMPRESSION: RUL atelectasis with Golden S sign.",
        "labels": "Atelectasis|Mass|Lung_Cancer",
    },
    {
        "diagnosis": "FINDINGS: Widespread platelike atelectasis in both lower lobes. Low lung volumes. Elevated hemidiaphragms. IMPRESSION: Bilateral basilar atelectasis due to hypoventilation.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Complete opacification of left hemithorax with mediastinal shift left. Right lung hyperinflated. IMPRESSION: Complete left lung atelectasis. Bronchoscopy recommended.",
        "labels": "Atelectasis|Mass",
    },
    {
        "diagnosis": "FINDINGS: Rounded atelectasis in right lower lobe with comet-tail sign. Adjacent pleural thickening. Stable. IMPRESSION: Rounded atelectasis.",
        "labels": "Atelectasis|Round_Atelectasis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Massive right pleural effusion with complete opacification of right hemithorax and contralateral mediastinal shift. IMPRESSION: Massive right pleural effusion.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Small right pleural effusion with intact meniscus sign. No pleural thickening. IMPRESSION: Small right pleural effusion.",
        "labels": "Effusion",
    },
    {
        "diagnosis": "FINDINGS: Bilateral pleural effusions with basal atelectasis. Cardiomegaly. Vascular congestion. IMPRESSION: Bilateral effusions in setting of CHF.",
        "labels": "Effusion|Cardiomegaly|Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Mild interstitial pulmonary edema with peribronchial cuffing, indistinct vascular margins, Kerley A and B lines. Mild cardiomegaly. IMPRESSION: Mild interstitial edema, early CHF.",
        "labels": "Edema|Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral airspace opacities more confluent centrally with peripheral sparing. No cardiomegaly. IMPRESSION: Noncardiogenic edema / ARDS.",
        "labels": "Edema|Infiltration|ARDS",
    },
    {
        "diagnosis": "FINDINGS: Asymmetric left-sided perihilar airspace opacities. No pleural effusion. Normal heart size. IMPRESSION: Asymmetric pulmonary edema.",
        "labels": "Edema|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Well-defined smoothly marginated nodule in left lower lobe with popcorn calcification. No growth. IMPRESSION: Hamartoma, benign.",
        "labels": "Nodule|Calcified_Granuloma",
    },
    {
        "diagnosis": "FINDINGS: Subsolid nodule with ground-glass and solid components (part-solid) in right middle lobe, 1.2 cm. IMPRESSION: Part-solid nodule, consider adenocarcinoma spectrum.",
        "labels": "Nodule|Lung_Cancer",
    },
    {
        "diagnosis": "FINDINGS: Numerous small well-defined nodules in perilymphatic distribution with fissural nodularity and right paratracheal adenopathy. IMPRESSION: Perilymphatic nodules, consider sarcoidosis.",
        "labels": "Nodule|Lymphadenopathy|Sarcoidosis",
    },
    {
        "diagnosis": "FINDINGS: Large 6 cm mass in left lower lobe with irregular borders and central necrosis. Left hilar adenopathy. IMPRESSION: Large left lower lobe mass, highly suspicious for malignancy.",
        "labels": "Mass|Nodule|Lung_Cancer|Lymphadenopathy",
    },
    {
        "diagnosis": "FINDINGS: Anterior mediastinal mass with lobulated contours. No calcification. Trachea midline. IMPRESSION: Anterior mediastinal mass, consider thymoma or lymphoma.",
        "labels": "Mass|Mediastinal_Mass",
    },
    {
        "diagnosis": "FINDINGS: Middle mediastinal mass causing splaying of the carina. IMPRESSION: Subcarinal mass, CT recommended.",
        "labels": "Mass|Mediastinal_Mass|Lymphadenopathy",
    },
    {
        "diagnosis": "FINDINGS: Bilateral upper lobe fibrotic changes with volume loss, hilar retraction, architectural distortion. Traction bronchiectasis. IMPRESSION: Chronic upper lobe fibrosis, post-TB or radiation.",
        "labels": "Fibrosis|Tuberculosis",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral ground-glass with reticular opacities and traction bronchiectasis in bases. No honeycombing. IMPRESSION: Probable ILD. HRCT recommended.",
        "labels": "Fibrosis|Infiltration|Interstitial_Lung_Disease",
    },
    {
        "diagnosis": "FINDINGS: Bilateral apical pleural thickening with subpleural fibrotic bands. Upper lobe volume loss. IMPRESSION: Chronic apical fibrosis, post-inflammatory.",
        "labels": "Fibrosis|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Crazy-paving: ground-glass opacities on interlobular septal thickening in bilateral lower lobes. IMPRESSION: Crazy-paving. Differential includes PAP, lipoid pneumonia.",
        "labels": "Infiltration|Fibrosis|Alveolar_Proteinosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral reticulonodular opacities mid-to-upper lung. Eggshell calcifications in hilar nodes. IMPRESSION: Silicosis with eggshell calcification.",
        "labels": "Fibrosis|Nodule|Silicosis|Pneumoconiosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral pleural plaques with calcification along diaphragmatic and lateral pleura. IMPRESSION: Asbestos-related pleural plaques. No mesothelioma.",
        "labels": "Pleural_Plaque|Pleural_Thickening|Asbestosis",
    },
    {
        "diagnosis": "FINDINGS: Diffuse fine reticulonodular opacities predominantly lower lobes. IMPRESSION: Early ILD, consider CTD-ILD or hypersensitivity pneumonitis.",
        "labels": "Fibrosis|Nodule|Interstitial_Lung_Disease",
    },
    {
        "diagnosis": "FINDINGS: Lymphangitic carcinomatosis: unilateral right septal thickening and peribronchial cuffing. Right hilar adenopathy. Small right effusion. IMPRESSION: Lymphangitic carcinomatosis.",
        "labels": "Infiltration|Effusion|Nodule|Lymphangitic_Carcinomatosis",
    },
    {
        "diagnosis": "FINDINGS: Left apical pleural thickening with fibrotic band to hilum. Calcified left hilar node. Traction bronchiectasis. IMPRESSION: Old TB with scarring.",
        "labels": "Fibrosis|Pleural_Thickening|Tuberculosis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral upper lobe fibrocavitary disease with thick-walled cavities. Elevated hila. IMPRESSION: Chronic fibrocavitary TB.",
        "labels": "Fibrosis|Infiltration|Cavitary_Lesion|Tuberculosis",
    },
    {
        "diagnosis": "FINDINGS: Cystic bronchiectasis in both lower lobes with thin-walled cysts and air-fluid levels. Signet-ring sign. Tree-in-bud. IMPRESSION: Cystic bronchiectasis with superinfection.",
        "labels": "Bronchiectasis|Tree_in_Bud|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Tram-track and ring-shadow opacities in right middle lobe and lingula. RML volume loss. IMPRESSION: RML and lingular bronchiectasis with volume loss.",
        "labels": "Bronchiectasis|Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Central bronchiectasis with dilated mucus-filled bronchi as glove-finger opacities from hila. Tree-in-bud. IMPRESSION: Central bronchiectasis, consider ABPA.",
        "labels": "Bronchiectasis|Tree_in_Bud|Mucus_Plugging",
    },
    {
        "diagnosis": "FINDINGS: Large hiatal hernia with mixed gas and soft tissue retrocardiac. Two air-fluid levels at different heights suggesting volvulus. IMPRESSION: Hiatal hernia with gastric volvulus.",
        "labels": "Hernia|Hiatal_Hernia",
    },
    {
        "diagnosis": "FINDINGS: Rigler sign: air on both sides of bowel wall. Free air under both hemidiaphragms. Visible falciform ligament. IMPRESSION: Massive pneumoperitoneum.",
        "labels": "Pneumoperitoneum",
    },
    {
        "diagnosis": "FINDINGS: Football sign in supine view: large oval lucency over upper abdomen. IMPRESSION: Large pneumoperitoneum (football sign).",
        "labels": "Pneumoperitoneum",
    },
    {
        "diagnosis": "FINDINGS: Mild cardiomegaly. Redistribution to upper lobes. No frank edema. No effusion. IMPRESSION: Mild cardiomegaly with early pulmonary venous hypertension.",
        "labels": "Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Globular cardiac silhouette severely enlarged. Pulmonary vascularity normal. Lungs clear. IMPRESSION: Severe cardiomegaly, consider pericardial effusion.",
        "labels": "Cardiomegaly|Pericardial_Effusion",
    },
    {
        "diagnosis": "FINDINGS: Moderate cardiomegaly with prominent left atrial appendage. Double density sign. Splayed carina. IMPRESSION: Cardiomegaly with LA enlargement, consider mitral valve disease.",
        "labels": "Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Epicardial fat pad sign with lucent line between heart and pericardium. Mild cardiomegaly. Lungs clear. IMPRESSION: Small-moderate pericardial effusion.",
        "labels": "Pericardial_Effusion",
    },
    {
        "diagnosis": "FINDINGS: Massive globular cardiomegaly with clear lungs. No vascular congestion. IMPRESSION: Large pericardial effusion, rule out tamponade.",
        "labels": "Pericardial_Effusion|Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Figure-of-3 sign in aortic knob with rib notching of posterior inferior ribs 3-8 bilaterally. IMPRESSION: Coarctation of the aorta.",
        "labels": "Coarctation",
    },
    {
        "diagnosis": "FINDINGS: Dilated azygos vein as comma-shaped density at right tracheobronchial angle. IMPRESSION: Prominent azygos vein, consider azygos continuation of IVC.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Egg-on-side cardiac silhouette with narrow vascular pedicle. Increased pulmonary vascularity. IMPRESSION: Transposition of the great arteries.",
        "labels": "Transposition_of_Great_Arteries|Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Boot-shaped heart (coeur en sabot) with upturned apex, prominent RV, concave PA segment. Decreased pulmonary vascularity. IMPRESSION: Tetralogy of Fallot.",
        "labels": "Tetralogy_of_F allot|Cardiomegaly",
    },
    {
        "diagnosis": "FINDINGS: Prominent thoracic aortic knob with calcification. Mediastinal width upper limits normal. IMPRESSION: Tortuous calcified aorta.",
        "labels": "Aortic_Aneurysm",
    },
    {
        "diagnosis": "FINDINGS: Wide mediastinum with loss of aortic knob contour. Tracheal deviation right. Left pleural effusion. IMPRESSION: Concerning for aortic dissection.",
        "labels": "Aortic_Dissection|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Prominent descending thoracic aorta 4.5 cm with calcified walls. IMPRESSION: Descending thoracic aortic aneurysm.",
        "labels": "Aortic_Aneurysm",
    },
    {
        "diagnosis": "FINDINGS: Calcified aortic arch aneurysm projecting right of trachea. IMPRESSION: Aortic arch aneurysm.",
        "labels": "Aortic_Aneurysm",
    },
    {
        "diagnosis": "FINDINGS: Bilateral diffuse fine nodular opacities mid-to-upper lung predominance. Minimal hilar adenopathy. IMPRESSION: Simple CWP.",
        "labels": "Nodule|Pneumoconiosis",
    },
    {
        "diagnosis": "FINDINGS: Multiple bilateral old rib fractures with callus formation. No acute fracture. IMPRESSION: Multiple old healed rib fractures.",
        "labels": "Rib_Fracture",
    },
    {
        "diagnosis": "FINDINGS: Sternal fracture with mild displacement on lateral view. No pneumothorax or hemothorax. No mediastinal widening. IMPRESSION: Sternal fracture.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral reticulonodular opacities with basal predominance. Bronchiectasis and architectural distortion. IMPRESSION: RA-associated ILD with UIP pattern.",
        "labels": "Fibrosis|Nodule|Interstitial_Lung_Disease|Bronchiectasis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral lower lobe ground-glass and reticular opacities with early honeycombing. Dilated esophagus with air-fluid level. IMPRESSION: Scleroderma-associated ILD with esophageal dilatation.",
        "labels": "Fibrosis|Infiltration|Interstitial_Lung_Disease",
    },
    {
        "diagnosis": "FINDINGS: Hyperlucent lungs with attenuation of peripheral vascular markings. Large central pulmonary arteries. Flattened diaphragms. IMPRESSION: Emphysema with pulmonary hypertension.",
        "labels": "Emphysema|COPD|Pulmonary_Hypertension",
    },
    {
        "diagnosis": "FINDINGS: Moderate hyperinflation with flattened diaphragms. Subtle reticular opacities in bases. IMPRESSION: COPD with mild interstitial changes.",
        "labels": "Emphysema|COPD|Fibrosis",
    },
    {
        "diagnosis": "FINDINGS: Hyperexpanded lungs with increased retrosternal airspace. Mild diaphragmatic flattening. IMPRESSION: Mild hyperinflation, early COPD.",
        "labels": "Emphysema|COPD",
    },
    {
        "diagnosis": "FINDINGS: Minimally displaced fracture of right lateral 7th rib. Small adjacent pleural effusion. IMPRESSION: Right 7th rib fracture with small effusion.",
        "labels": "Rib_Fracture|Effusion",
    },
    {
        "diagnosis": "FINDINGS: Multiple left rib fractures 4-8 with flail segment. Large left hemothorax. Mediastinal shift right. IMPRESSION: Left flail chest with massive hemothorax.",
        "labels": "Rib_Fracture|Effusion|Hemothorax",
    },
    {
        "diagnosis": "FINDINGS: Fracture of left clavicle with inferior displacement. No pneumothorax. Lungs clear. IMPRESSION: Isolated left clavicular fracture.",
        "labels": "",
    },
    {
        "diagnosis": "FINDINGS: Superior sulcus opacity on right with thickened pleura and erosion of posterior right first rib. IMPRESSION: Pancoast tumor with rib destruction.",
        "labels": "Mass|Pancoast_Tumor",
    },
    {
        "diagnosis": "FINDINGS: HIPAA chest tube in right hemithorax with tip in pleural space. Right lung re-expanded. Small residual effusion. IMPRESSION: Chest tube in place. Improving pneumothorax/effusion.",
        "labels": "Support_Devices|Post_Surgical_Changes",
    },
    {
        "diagnosis": "FINDINGS: Endotracheal tube tip 3 cm above carina. Nasogastric tube in stomach. Central venous catheter tip in SVC. Bilateral lung expansion adequate. IMPRESSION: Lines and tubes in satisfactory position.",
        "labels": "Support_Devices",
    },
    {
        "diagnosis": "FINDINGS: Cardiac pacemaker with leads in right atrium and right ventricle. Lungs clear. Heart size normal. IMPRESSION: Pacemaker in good position. Normal chest.",
        "labels": "Support_Devices",
    },
    {
        "diagnosis": "FINDINGS: Postoperative changes: median sternotomy wires. Bilateral lung expansion good. Small left pleural effusion. No pneumothorax. IMPRESSION: Post-CABG with stable findings.",
        "labels": "Post_Surgical_Changes|Effusion|Support_Devices",
    },
    {
        "diagnosis": "FINDINGS: Right-sided port-a-cath with tip in the superior vena cava. Lungs clear. No pneumothorax. IMPRESSION: Port-a-cath in satisfactory position.",
        "labels": "Support_Devices",
    },
    {
        "diagnosis": "FINDINGS: Bilateral interstitial and airspace opacities with peripheral distribution and eosinophilia suggested by history. Normal heart size. IMPRESSION: Consider eosinophilic pneumonia.",
        "labels": "Consolidation|Infiltration",
    },
    {
        "diagnosis": "FINDINGS: Hyperinflation with increased AP diameter. Flattened diaphragms. No acute infiltrate. IMPRESSION: COPD, no acute cardiopulmonary disease.",
        "labels": "COPD|Emphysema",
    },
    {
        "diagnosis": "FINDINGS: Eventration of the right hemidiaphragm with elevated dome. Lungs clear. Heart size normal. IMPRESSION: Right diaphragmatic eventration.",
        "labels": "Eventration",
    },
    {
        "diagnosis": "FINDINGS: Kyphoscoliosis of the thoracic spine with associated rotation and asymmetric lung volumes. Heart size normal. Lungs clear. IMPRESSION: Thoracic kyphoscoliosis.",
        "labels": "Kyphoscoliosis",
    },
    {
        "diagnosis": "FINDINGS: Diffuse interstitial lung disease with honeycombing and traction bronchiectasis. No acute consolidation. IMPRESSION: End-stage interstitial lung disease.",
        "labels": "Fibrosis|Interstitial_Lung_Disease|Bronchiectasis",
    },
    {
        "diagnosis": "FINDINGS: Bilateral pleural thickening with calcified pleural plaques. No pleural effusion. Lungs clear. IMPRESSION: Chronic pleural thickening due to asbestos exposure.",
        "labels": "Pleural_Thickening|Pleural_Plaque|Asbestosis",
    },
    {
        "diagnosis": "FINDINGS: Large bullae in the right upper lobe occupying approximately one-third of the hemithorax. Compression of adjacent lung. No pneumothorax. IMPRESSION: Large right upper lobe bulla.",
        "labels": "Emphysema|COPD",
    },
    {
        "diagnosis": "FINDINGS: Ankylosing spondylitis with bamboo spine and fusion of the costovertebral joints. Restrictive lung physiology suggested by low lung volumes. Lungs clear. IMPRESSION: Ankylosing spondylitis with thoracic involvement.",
        "labels": "Ankylosing_Spondylitis",
    },
    {
        "diagnosis": "FINDINGS: Right middle lobe atelectasis with silhouette sign against right heart border. No effusion. Heart size normal. IMPRESSION: RML atelectasis.",
        "labels": "Atelectasis",
    },
    {
        "diagnosis": "FINDINGS: Left upper lobe cavitary lesion with thick irregular walls. Surrounding consolidation. No air-fluid level. IMPRESSION: Cavitary left upper lobe lesion, differential includes TB, fungal, or malignancy.",
        "labels": "Cavitary_Lesion|Mass|Tuberculosis",
    },
    {
        "diagnosis": "FINDINGS: Extensive unilateral right-sided pleural effusion with loculation and pleural thickening. Underlying lung partially obscured. IMPRESSION: Complicated pleural effusion/empyema.",
        "labels": "Effusion|Empyema|Pleural_Thickening",
    },
    {
        "diagnosis": "FINDINGS: Bilateral hilar lymphadenopathy with normal underlying lungs. No pleural effusion. No pneumothorax. IMPRESSION: Bilateral hilar lymphadenopathy. Consider sarcoidosis, TB, or lymphoma.",
        "labels": "Lymphadenopathy|Sarcoidosis",
    },
    {
        "diagnosis": "FINDINGS: Diffuse bilateral alveolar filling process with air bronchograms and air alveolograms. Normal heart size. No effusion. IMPRESSION: Diffuse alveolar filling process, consider alveolar hemorrhage or diffuse alveolar damage.",
        "labels": "Consolidation|Alveolar_Hemorrhage|ARDS",
    },
    {
        "diagnosis": "FINDINGS: Right paratracheal mass with smooth borders. No calcification. Trachea slightly deviated left. No pleural effusion. IMPRESSION: Right paratracheal mass. CT for further characterization.",
        "labels": "Mass|Mediastinal_Mass",
    },
    {
        "diagnosis": "FINDINGS: Left apical cap with associated left upper lobe volume loss. No definite mass. No effusion. IMPRESSION: Left apical cap. Consider Pancoast tumor versus benign pleural thickening.",
        "labels": "Pleural_Thickening|Pancoast_Tumor",
    },
]

EXISTING_IMAGES_CACHE = None


def get_existing_images():
    global EXISTING_IMAGES_CACHE
    if EXISTING_IMAGES_CACHE is not None:
        return EXISTING_IMAGES_CACHE
    images = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = row.get("image_path", "").strip()
                if p and p not in images:
                    images.append(p)
    if not images:
        images = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
                  if f.endswith((".jpg", ".jpeg", ".png"))]
    EXISTING_IMAGES_CACHE = images
    return images


def load_augmented(max_entries=None):
    print(f"\n{'='*60}")
    print(f"LOADING: Augmented rare findings ({len(AUGMENTED_ENTRIES)} entries)")
    print(f"{'='*60}")
    images = get_existing_images()
    if not images:
        print("  No existing images to pair with augmented entries. Skipping.")
        return 0

    entries = AUGMENTED_ENTRIES[:max_entries] if max_entries else AUGMENTED_ENTRIES
    written = 0
    for i, entry in enumerate(entries):
        img_path = random.choice(images)
        symptoms = random.choice(SYMPTOM_TEMPLATES)
        if "{days}" in symptoms:
            symptoms = symptoms.format(days=random.choice([3, 5, 7, 10, 14]))
        append(img_path, "augmented", symptoms, entry["diagnosis"], entry["labels"])
        written += 1
    print(f"  Augmented entries loaded: {written}")
    return written


def main():
    pre_count = count_existing()
    print(f"Existing dataset: {pre_count} rows")

    write_header()

    total = pre_count

    total += load_chexpert(max_images=50000)
    total += load_padchest(max_images=40000)
    total += load_vindr(max_images=18000)
    total += load_chexphoto(max_images=10000)
    total += load_siim_pneumothorax(max_images=12000)
    total += load_rsna_pneumonia(max_images=30000)
    total += load_nih_full(max_images=80000)
    total += load_augmented()

    print(f"\n{'='*60}")
    print(f"EXPANSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Previous rows: {pre_count}")
    print(f"  New rows:      {total - pre_count}")
    print(f"  Total rows:    {total}")

    img_count = len([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".jpeg", ".png"))])
    print(f"  Images in dir: {img_count}")

    csv_size = os.path.getsize(CSV_PATH) / (1024 * 1024) if os.path.exists(CSV_PATH) else 0
    print(f"  CSV size:      {csv_size:.1f} MB")

    print(f"\nNext: Train on this expanded data:")
    print(f"  python training_fusion.py --mode train --epochs 20 --batch_size 16 --use_amp --mode multilabel --wider_head")
    print(f"\nOr upload to GCS for Vertex AI training:")
    print(f"  gsutil cp {CSV_PATH} gs://YOUR_BUCKET/data/")
    print(f"  gsutil -m cp {IMAGES_DIR}\\*.jpg gs://YOUR_BUCKET/data/images/")


if __name__ == "__main__":
    main()