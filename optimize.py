import gc
import os
import threading

import psutil
import torch
from PIL import Image

MODEL_DIR = "./blip-xray-finetuned"
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fusion_model.pth")
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")
DEFAULT_MODEL_DIR = os.path.join("models", "default")
DEFAULT_CLASSIFIER_ONNX = os.path.join(DEFAULT_MODEL_DIR, "fusion_classifier.onnx")
DEFAULT_LABELS_PATH = os.path.join(DEFAULT_MODEL_DIR, "labels.json")
ONNX_FULL_DIR = os.path.join(CHECKPOINT_DIR, "onnx_full")
ONNX_FULL_PATH = os.path.join(ONNX_FULL_DIR, "fusion_full.onnx")
ONNX_FULL_LABELS = os.path.join(ONNX_FULL_DIR, "labels.json")

NIH_LABELS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

_loaded_blip = None
_loaded_fusion = None

# ── CPU Thread Control ────────────────────────────────────────

def set_cpu_threads(n=None):
    if n is None:
        n = max(1, psutil.cpu_count(logical=True) // 2)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    return n

# ── Memory ────────────────────────────────────────────────────

def get_memory_usage():
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "rss_mb": mem.rss / 1024 / 1024,
        "vms_mb": mem.vms / 1024 / 1024,
    }

def clear_memory():
    global _loaded_blip, _loaded_fusion
    _loaded_blip = None
    _loaded_fusion = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ── Device ─────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def use_fp16():
    return get_device() in ("cuda", "mps")

# ── BLIP / Vision inference ───────────────────────────────────

def infer_blip(image_path, use_onnx=True):
    global _loaded_blip
    image = Image.open(image_path).convert("RGB")

    if use_onnx and os.path.exists(os.path.join(ONNX_DIR, "model.onnx")):
        return _infer_blip_onnx(image)
    return _infer_blip_pytorch(image)

def _infer_blip_pytorch(image):
    global _loaded_blip
    if _loaded_blip is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_name = MODEL_DIR if os.path.exists(MODEL_DIR) else "Salesforce/blip-image-captioning-base"
        _loaded_blip = {
            "processor": BlipProcessor.from_pretrained(model_name),
            "model": BlipForConditionalGeneration.from_pretrained(model_name).eval(),
        }
        if use_fp16() and hasattr(torch, "float16"):
            try:
                _loaded_blip["model"] = _loaded_blip["model"].half()
            except Exception:
                pass

    p, m = _loaded_blip["processor"], _loaded_blip["model"]
    inputs = p(images=image, return_tensors="pt")
    if use_fp16():
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        out = m.generate(**inputs, max_length=64)
    return p.decode(out[0], skip_special_tokens=True)

def _infer_blip_onnx(image):
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import BlipProcessor
    processor = BlipProcessor.from_pretrained(ONNX_DIR)
    model = ORTModelForVision2Seq.from_pretrained(ONNX_DIR, provider="CPUExecutionProvider")
    inputs = processor(images=image, return_tensors="np")
    out = model.generate(**inputs, max_length=64)
    return processor.decode(out[0], skip_special_tokens=True)

# ── Fusion / Symptom Check inference ──────────────────────────

def get_available_models():
    """Return a dict describing which models are available."""
    return {
        "trained_pytorch": os.path.exists(CHECKPOINT_PATH),
        "default_classifier": os.path.exists(DEFAULT_CLASSIFIER_ONNX),
        "onnx_full_pipeline": os.path.exists(ONNX_FULL_PATH),
    }


def _infer_fusion_default(image_path, symptoms):
    """Fallback: use default ONNX classifier with stock PyTorch encoders."""
    global _loaded_fusion
    if _loaded_fusion is None:
        import json
        from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
        from training import DiagnosisFusionModel

        label_list = NIH_LABELS
        model = DiagnosisFusionModel(num_conditions=len(label_list))
        # Reset classifier to random weights if no checkpoint
        if not os.path.exists(CHECKPOINT_PATH):
            for layer in model.classifier:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        _loaded_fusion = {
            "model": model.eval(),
            "label_list": label_list,
        }

    image = Image.open(image_path).convert("RGB")
    m = _loaded_fusion["model"]
    label_list = _loaded_fusion["label_list"]

    with torch.no_grad():
        logits = m([image], [symptoms])
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)

    return label_list[predicted.item()], confidence.item()


def infer_fusion(image_path, symptoms):
    global _loaded_fusion

    # Priority 1: Trained PyTorch model
    if _loaded_fusion is None and os.path.exists(CHECKPOINT_PATH):
        from training import DiagnosisFusionModel
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        label_list = checkpoint.get("label_list", [])
        if label_list:
            model = DiagnosisFusionModel(num_conditions=len(label_list))
            model.classifier.load_state_dict(checkpoint["model_state"])
            _loaded_fusion = {
                "model": model.eval(),
                "label_list": label_list,
            }

    # Priority 2: Default model (stock encoders + fresh classifier)
    if _loaded_fusion is None:
        return _infer_fusion_default(image_path, symptoms)

    image = Image.open(image_path).convert("RGB")
    m = _loaded_fusion["model"]
    label_list = _loaded_fusion["label_list"]

    with torch.no_grad():
        logits = m([image], [symptoms])
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)

    return label_list[predicted.item()], confidence.item()

# ── Full ONNX Pipeline Export ─────────────────────────────────

def _ensure_onnx_deps():
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        from rich.console import Console
        console = Console()
        console.print("[yellow]Installing ONNX dependencies...[/yellow]")
        import subprocess, sys
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "onnx", "onnxruntime", "onnxscript",
        ])
        return True

def _load_fusion_model():
    from training import DiagnosisFusionModel
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    label_list = checkpoint.get("label_list", [])
    if not label_list:
        return None, None
    model = DiagnosisFusionModel(num_conditions=len(label_list))
    model.classifier.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, label_list

class _FullFusionONNXWrapper(torch.nn.Module):
    """Wraps the full fusion pipeline so torch.onnx.export can trace it end-to-end."""
    def __init__(self, model):
        super().__init__()
        self.image_encoder = model.image_encoder.vision_model
        self.symptom_encoder = model.symptom_encoder
        self.classifier = model.classifier
        self.image_proj = model.image_encoder.visual_projection

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.image_encoder(pixel_values)
        image_features = self.image_proj(vision_outputs.pooler_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_outputs = self.symptom_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state.mean(dim=1)

        combined = torch.cat([image_features, text_features], dim=-1)
        return self.classifier(combined)

def export_full_fusion_onnx(output_dir=None):
    """Export the entire fusion pipeline (image + text -> logits) to a single ONNX file."""
    if output_dir is None:
        output_dir = os.path.join(CHECKPOINT_DIR, "onnx_full")
    os.makedirs(output_dir, exist_ok=True)

    from rich.console import Console
    console = Console()

    model, label_list = _load_fusion_model()
    if model is None:
        console.print("[red]No model or labels found. Train first.[/red]")
        return

    _ensure_onnx_deps()

    wrapper = _FullFusionONNXWrapper(model).eval()

    dummy_pixel = torch.randn(1, 3, 224, 224)
    dummy_ids = torch.randint(0, 100, (1, 64), dtype=torch.long)
    dummy_mask = torch.ones(1, 64, dtype=torch.long)

    console.print("[cyan]Exporting full fusion pipeline to ONNX...[/cyan]")

    torch.onnx.export(
        wrapper,
        (dummy_pixel, dummy_ids, dummy_mask),
        os.path.join(output_dir, "fusion_full.onnx"),
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    import json
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(label_list, f)

    console.print(f"[green]Full ONNX model saved to {output_dir}/fusion_full.onnx[/green]")
    console.print(f"[green]Labels saved to {output_dir}/labels.json[/green]")
    console.print(f"[green]Model has {len(label_list)} output classes.[/green]")

def infer_fusion_onnx(image_path, symptoms, model_dir=None):
    """Run inference using the full ONNX pipeline. No PyTorch needed beyond preprocessing.

    Searches for models in this order:
      1. checkpoints/onnx_full/  (trained full pipeline)
      2. models/default/         (default shipped classifier)
    """
    import json
    import numpy as np
    import onnxruntime as ort
    from transformers import CLIPProcessor, AutoTokenizer

    # Find the best available ONNX model + labels
    candidates = [
        (model_dir, "fusion_full.onnx", "labels.json"),
        (ONNX_FULL_DIR, "fusion_full.onnx", "labels.json"),
        (DEFAULT_MODEL_DIR, "fusion_classifier.onnx", "labels.json"),
    ]

    onnx_path = None
    labels_path = None
    for d, m, l in candidates:
        if d is None:
            continue
        mp = os.path.join(d, m)
        lp = os.path.join(d, l)
        if os.path.exists(mp) and os.path.exists(lp):
            onnx_path = mp
            labels_path = lp
            break

    if onnx_path is None:
        return None, "No ONNX model found. Run 'python setup_default.py' or 'python quantization.py --mode export-full'."

    with open(labels_path) as f:
        label_list = json.load(f)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    image = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=image, return_tensors="np")
    pixel_values = img_inputs["pixel_values"].astype(np.float32)

    tok_inputs = tokenizer(symptoms, return_tensors="np", padding="max_length", truncation=True, max_length=64)
    input_ids = tok_inputs["input_ids"].astype(np.int64)
    attention_mask = tok_inputs["attention_mask"].astype(np.int64)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logits = session.run(None, {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0]

    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    predicted = np.argmax(probs, axis=-1)
    confidence = float(probs[0, predicted[0]])

    return label_list[predicted[0]], confidence

# ── Legacy Quantization (kept for backward compat) ──────────

def quantize_blip(output_dir=None):
    from rich.console import Console
    console = Console()
    console.print("[yellow]BLIP ONNX quantization requires a newer version of optimum.[/yellow]")
    console.print("[yellow]Run: pip install --upgrade optimum[/yellow]")
    console.print("[yellow]The system uses PyTorch automatically until then.[/yellow]")

def quantize_fusion(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(CHECKPOINT_DIR, "onnx")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(CHECKPOINT_PATH):
        print("No fusion checkpoint found. Train first.")
        return

    from rich.console import Console
    console = Console()

    try:
        import onnxscript
    except ImportError:
        console.print("[yellow]'onnxscript' is required for ONNX export.[/yellow]")
        import questionary
        if questionary.confirm("Install onnxscript now?", default=True).ask():
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxscript"])
        else:
            console.print("[yellow]Skipped. The system works fine without ONNX export.[/yellow]")
            return

    console.print("[cyan]The fusion model uses frozen CLIP + BERT encoders.[/cyan]")
    console.print("[cyan]Exporting the classifier head only to ONNX (encoders stay in PyTorch).[/cyan]")

    from training import DiagnosisFusionModel, load_label_list

    label_list = load_label_list()
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = DiagnosisFusionModel(num_conditions=len(label_list))
    model.classifier.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 512 + 768)
    torch.onnx.export(
        model.classifier,
        dummy,
        os.path.join(output_dir, "fusion_classifier.onnx"),
        input_names=["features"],
        output_names=["logits"],
        opset_version=14,
    )
    console.print(f"[green]  ONNX classifier saved to {output_dir}[/green]")
