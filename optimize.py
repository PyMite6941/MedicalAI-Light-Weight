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
FUSION_ONNX_DIR = os.path.join(CHECKPOINT_DIR, "onnx")
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
_fusion_encoders = None  # CLIP+BERT encoders, cached for the classifier-only ONNX path

# ── CPU Thread Control ────────────────────────────────────────
# Threads are capped at half the logical cores everywhere (torch, ONNX
# Runtime, MKL/OpenMP libs) so the app stays a well-behaved background
# citizen on shared / low-spec machines instead of saturating every core.

_cpu_thread_count = None

def set_cpu_threads(n=None):
    global _cpu_thread_count
    if n is None:
        n = max(1, psutil.cpu_count(logical=True) // 2)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    _cpu_thread_count = n
    return n

def get_cpu_threads():
    return _cpu_thread_count or set_cpu_threads()

# ── Memory ────────────────────────────────────────────────────

def get_memory_usage():
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        "rss_mb": mem.rss / 1024 / 1024,
        "vms_mb": mem.vms / 1024 / 1024,
    }

def clear_memory():
    """Drop cached models/sessions and force garbage collection.

    Called after every inference request so the process gives memory back
    between requests instead of holding every model it has ever touched.
    """
    global _loaded_blip, _loaded_fusion, _fusion_encoders
    _loaded_blip = None
    _loaded_fusion = None
    _fusion_encoders = None
    _onnx_session_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ── Device Detection: TPU > CUDA (NVIDIA GPU) > MPS (Apple GPU) > CPU ──
#
# Every accelerator class is probed cheaply and in order of how much
# horsepower it typically offers, with CPU as the universal fallback so
# the app always runs somewhere. The result is cached — hardware doesn't
# change mid-process, and re-probing (especially the TPU check) has a
# real cost.

_device_cache = None

def _tpu_available():
    """Best-effort TPU detection via torch_xla.

    Only attempts the (relatively expensive) torch_xla import when a TPU
    runtime env var is actually present (set by Colab/Kaggle/GCP TPU VMs),
    so machines without a TPU never pay for the probe.
    """
    if not (os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR") or os.environ.get("XRT_TPU_CONFIG")):
        return False
    try:
        import torch_xla.core.xla_model as xm
        xm.xla_device()
        return True
    except Exception:
        return False

def get_device():
    """Return the best available compute device: 'tpu', 'cuda', 'mps', or 'cpu'."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if _tpu_available():
        device = "tpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    _device_cache = device
    return device

def get_torch_device():
    """Return a torch.device (or XLA device) matching get_device()."""
    device = get_device()
    if device == "tpu":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device(device)

def use_fp16():
    return get_device() in ("cuda", "mps")

def device_summary():
    """Diagnostics dict for status/health endpoints."""
    return {
        "device": get_device().upper(),
        "fp16": use_fp16(),
        "cpu_threads": get_cpu_threads(),
        "onnx_providers": get_onnx_providers(),
    }

# ── ONNX Runtime: providers + thread-capped session options ─────

_onnx_providers_cache = None

def get_onnx_providers():
    """Ordered ONNX Runtime execution providers matching the detected device.

    Only returns providers actually compiled into the installed
    onnxruntime build, always ending in CPUExecutionProvider so inference
    never has nowhere to run.
    """
    global _onnx_providers_cache
    if _onnx_providers_cache is not None:
        return _onnx_providers_cache

    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except ImportError:
        _onnx_providers_cache = ["CPUExecutionProvider"]
        return _onnx_providers_cache

    device = get_device()

    priority = []
    if device == "cuda":
        priority += ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    priority += ["ROCMExecutionProvider"]  # AMD GPUs; no-op if unavailable
    if device == "mps":
        priority += ["CoreMLExecutionProvider"]
    priority += ["DmlExecutionProvider"]  # Windows DirectML GPU fallback
    priority += ["CPUExecutionProvider"]

    providers = [p for p in priority if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    _onnx_providers_cache = providers
    return providers

def get_ort_session_options():
    """SessionOptions capped to the same thread budget as set_cpu_threads().

    Without this, ONNX Runtime defaults to using every logical core for
    intra-op parallelism regardless of what the rest of the app is doing.
    """
    import onnxruntime as ort
    n = get_cpu_threads()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n
    opts.inter_op_num_threads = max(1, n // 2)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    return opts

_onnx_session_cache = {}

def _get_onnx_session(onnx_path):
    """Cached InferenceSession keyed by path. Only one session is kept
    resident at a time — loading a new one evicts the previous — so
    switching between models doesn't accumulate memory."""
    if onnx_path not in _onnx_session_cache:
        import onnxruntime as ort
        _onnx_session_cache.clear()
        _onnx_session_cache[onnx_path] = ort.InferenceSession(
            onnx_path,
            sess_options=get_ort_session_options(),
            providers=get_onnx_providers(),
        )
    return _onnx_session_cache[onnx_path]

# ── INT8 Dynamic Quantization ────────────────────────────────────

def quantize_onnx_model(input_path, output_path=None, weight_type="quint8"):
    """Apply INT8 dynamic quantization to an ONNX model for a smaller,
    faster-on-CPU model. Returns the output path, or None if quantization
    isn't available / didn't succeed (caller should fall back to the
    full-precision model in that case, never hard-fail)."""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}.int8{ext}"

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        return None

    qtype = QuantType.QUInt8 if weight_type == "quint8" else QuantType.QInt8
    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=qtype,
        )
    except Exception as e:
        try:
            from rich.console import Console
            Console().print(f"[yellow]INT8 quantization skipped for {input_path}: {e}[/yellow]")
        except Exception:
            print(f"INT8 quantization skipped for {input_path}: {e}")
        return None

    return output_path

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
        device = get_torch_device()
        model = BlipForConditionalGeneration.from_pretrained(model_name).eval()
        # generate() isn't XLA-graph-friendly without torch_xla-specific handling,
        # so TPUs run BLIP on CPU here rather than risk a broken generation loop.
        if get_device() != "tpu":
            model = model.to(device)
        _loaded_blip = {
            "processor": BlipProcessor.from_pretrained(model_name),
            "model": model,
            "device": device,
        }
        if use_fp16() and hasattr(torch, "float16"):
            try:
                _loaded_blip["model"] = _loaded_blip["model"].half()
            except Exception:
                pass

    p, m, device = _loaded_blip["processor"], _loaded_blip["model"], _loaded_blip["device"]
    inputs = p(images=image, return_tensors="pt")
    if get_device() != "tpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    if use_fp16():
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    with torch.no_grad():
        out = m.generate(**inputs, max_length=64)
    return p.decode(out[0], skip_special_tokens=True)

def _infer_blip_onnx(image):
    from optimum.onnxruntime import ORTModelForVision2Seq
    from transformers import BlipProcessor
    processor = BlipProcessor.from_pretrained(ONNX_DIR)
    provider = get_onnx_providers()[0]
    model = ORTModelForVision2Seq.from_pretrained(ONNX_DIR, provider=provider)
    inputs = processor(images=image, return_tensors="np")
    out = model.generate(**inputs, max_length=64)
    return processor.decode(out[0], skip_special_tokens=True)

# ── Fusion / Symptom Check inference ──────────────────────────

def get_available_models():
    """Return a dict describing which models are available."""
    onnx_full = os.path.exists(ONNX_FULL_PATH) or os.path.exists(
        os.path.join(ONNX_FULL_DIR, "fusion_full.int8.onnx")
    )
    onnx_classifier = os.path.exists(os.path.join(FUSION_ONNX_DIR, "fusion_classifier.onnx")) or os.path.exists(
        os.path.join(FUSION_ONNX_DIR, "fusion_classifier.int8.onnx")
    )
    onnx_quantized = any(
        os.path.exists(p) for p in (
            os.path.join(ONNX_FULL_DIR, "fusion_full.int8.onnx"),
            os.path.join(FUSION_ONNX_DIR, "fusion_classifier.int8.onnx"),
            os.path.join(DEFAULT_MODEL_DIR, "fusion_full.int8.onnx"),
            os.path.join(DEFAULT_MODEL_DIR, "fusion_classifier.int8.onnx"),
        )
    )
    return {
        "trained_pytorch": os.path.exists(CHECKPOINT_PATH),
        "default_classifier": os.path.exists(DEFAULT_CLASSIFIER_ONNX),
        "onnx_full_pipeline": onnx_full,
        "onnx_classifier": onnx_classifier,
        "onnx_quantized": onnx_quantized,
    }


def _infer_fusion_default(image_path, symptoms):
    """Fallback: use default ONNX classifier with stock PyTorch encoders."""
    global _loaded_fusion
    if _loaded_fusion is None:
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

def export_full_fusion_onnx(output_dir=None, quantize=True):
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

    onnx_path = os.path.join(output_dir, "fusion_full.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_pixel, dummy_ids, dummy_mask),
        onnx_path,
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

    console.print(f"[green]Full ONNX model saved to {onnx_path}[/green]")
    console.print(f"[green]Labels saved to {output_dir}/labels.json[/green]")
    console.print(f"[green]Model has {len(label_list)} output classes.[/green]")

    if quantize:
        console.print("[cyan]Applying INT8 dynamic quantization...[/cyan]")
        int8_path = quantize_onnx_model(onnx_path)
        if int8_path:
            orig_mb = os.path.getsize(onnx_path) / 1024 / 1024
            new_mb = os.path.getsize(int8_path) / 1024 / 1024
            console.print(f"[green]Quantized model saved to {int8_path} ({orig_mb:.1f} MB -> {new_mb:.1f} MB)[/green]")
        else:
            console.print("[yellow]Quantization skipped (onnxruntime not installed or model incompatible).[/yellow]")

def _infer_fusion_full_onnx(onnx_path, labels_path, image_path, symptoms):
    """Run inference through a single full-pipeline ONNX file: image + text -> logits.
    No PyTorch needed beyond preprocessing (CLIP/BERT preprocessors)."""
    import json
    import numpy as np
    from transformers import CLIPProcessor, AutoTokenizer

    with open(labels_path) as f:
        label_list = json.load(f)

    pre = _get_onnx_preprocessors()
    clip_processor, tokenizer = pre["clip_processor"], pre["tokenizer"]

    image = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(images=image, return_tensors="np")
    pixel_values = img_inputs["pixel_values"].astype(np.float32)

    tok_inputs = tokenizer(symptoms, return_tensors="np", padding="max_length", truncation=True, max_length=64)
    input_ids = tok_inputs["input_ids"].astype(np.int64)
    attention_mask = tok_inputs["attention_mask"].astype(np.int64)

    session = _get_onnx_session(onnx_path)
    logits = session.run(None, {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })[0]

    return _softmax_top1(logits, label_list)

def _infer_fusion_classifier_onnx(onnx_path, labels_path, image_path, symptoms):
    """Run inference through a classifier-only ONNX head, using cached PyTorch
    CLIP/BERT encoders to produce the 1280-dim feature vector it expects."""
    global _fusion_encoders
    import json
    import numpy as np

    with open(labels_path) as f:
        label_list = json.load(f)

    if _fusion_encoders is None:
        from training import DiagnosisFusionModel
        _fusion_encoders = DiagnosisFusionModel(num_conditions=1).eval()

    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        image_vec = _fusion_encoders.encode_images([image])
        symptom_vec = _fusion_encoders.encode_symptoms([symptoms])
        features = torch.cat([image_vec, symptom_vec], dim=-1).numpy().astype(np.float32)

    session = _get_onnx_session(onnx_path)
    logits = session.run(None, {"features": features})[0]

    return _softmax_top1(logits, label_list)

def _softmax_top1(logits, label_list):
    import numpy as np
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    predicted = np.argmax(probs, axis=-1)
    confidence = float(probs[0, predicted[0]])
    return label_list[predicted[0]], confidence

_onnx_preprocessors_cache = None

def _get_onnx_preprocessors():
    """CLIP image processor + BERT tokenizer, loaded once and reused across
    every ONNX inference call instead of hitting the HF cache every time."""
    global _onnx_preprocessors_cache
    if _onnx_preprocessors_cache is None:
        from transformers import CLIPProcessor, AutoTokenizer
        _onnx_preprocessors_cache = {
            "clip_processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            "tokenizer": AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT"),
        }
    return _onnx_preprocessors_cache

def infer_fusion_onnx(image_path, symptoms, model_dir=None):
    """Run inference using the best available ONNX model, INT8-quantized first.

    Searches in this order:
      1. Full pipeline (image+text -> logits), no PyTorch needed:
         <model_dir>/, checkpoints/onnx_full/, models/default/
      2. Classifier-only head + PyTorch CLIP/BERT encoders:
         checkpoints/onnx/, models/default/
    Quantized (*.int8.onnx) variants are preferred over full precision.
    """
    full_dirs = [d for d in (model_dir, ONNX_FULL_DIR, DEFAULT_MODEL_DIR) if d]
    for d in full_dirs:
        for fname in ("fusion_full.int8.onnx", "fusion_full.onnx"):
            onnx_path = os.path.join(d, fname)
            labels_path = os.path.join(d, "labels.json")
            if os.path.exists(onnx_path) and os.path.exists(labels_path):
                return _infer_fusion_full_onnx(onnx_path, labels_path, image_path, symptoms)

    classifier_dirs = [d for d in (model_dir, FUSION_ONNX_DIR, DEFAULT_MODEL_DIR) if d]
    for d in classifier_dirs:
        for fname in ("fusion_classifier.int8.onnx", "fusion_classifier.onnx"):
            onnx_path = os.path.join(d, fname)
            labels_path = os.path.join(d, "labels.json")
            if os.path.exists(onnx_path) and os.path.exists(labels_path):
                return _infer_fusion_classifier_onnx(onnx_path, labels_path, image_path, symptoms)

    return None, "No ONNX model found. Run 'python setup_default.py' or 'python quantization.py --mode export-full'."

# ── Quantization ──────────────────────────────────────────────

def quantize_blip(output_dir=None, quantize=True):
    """Export BLIP (fine-tuned if available, else stock) to ONNX via optimum,
    then apply INT8 dynamic quantization for fast, standalone CPU inference."""
    if output_dir is None:
        output_dir = ONNX_DIR
    os.makedirs(output_dir, exist_ok=True)

    from rich.console import Console
    console = Console()

    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import BlipProcessor
    except ImportError:
        console.print("[yellow]'optimum[onnxruntime]' is required for BLIP ONNX export.[/yellow]")
        console.print("[yellow]Run: pip install --upgrade 'optimum[onnxruntime]'[/yellow]")
        return

    model_name = MODEL_DIR if os.path.exists(MODEL_DIR) else "Salesforce/blip-image-captioning-base"
    console.print(f"[cyan]Exporting BLIP ({model_name}) to ONNX...[/cyan]")
    try:
        model = ORTModelForVision2Seq.from_pretrained(model_name, export=True)
        processor = BlipProcessor.from_pretrained(model_name)
    except Exception as e:
        console.print(f"[red]BLIP ONNX export failed: {e}[/red]")
        console.print("[yellow]The system uses PyTorch automatically until this is resolved.[/yellow]")
        return

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    console.print(f"[green]BLIP ONNX saved to {output_dir}[/green]")

    if not quantize:
        return

    console.print("[cyan]Applying INT8 dynamic quantization to BLIP components...[/cyan]")
    quantized_any = False
    for fname in os.listdir(output_dir):
        if fname.endswith(".onnx") and not fname.endswith(".int8.onnx"):
            src = os.path.join(output_dir, fname)
            result = quantize_onnx_model(src)
            if result:
                quantized_any = True
    if quantized_any:
        console.print(f"[green]Quantized BLIP components saved alongside full-precision ones in {output_dir}[/green]")
    else:
        console.print("[yellow]Quantization skipped for BLIP components (onnxruntime not installed).[/yellow]")

def quantize_fusion(output_dir=None, quantize=True):
    if output_dir is None:
        output_dir = FUSION_ONNX_DIR
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
    onnx_path = os.path.join(output_dir, "fusion_classifier.onnx")
    torch.onnx.export(
        model.classifier,
        dummy,
        onnx_path,
        input_names=["features"],
        output_names=["logits"],
        opset_version=14,
    )
    import json
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(label_list, f)
    console.print(f"[green]  ONNX classifier saved to {output_dir}[/green]")

    if quantize:
        console.print("[cyan]  Applying INT8 dynamic quantization...[/cyan]")
        int8_path = quantize_onnx_model(onnx_path)
        if int8_path:
            orig_kb = os.path.getsize(onnx_path) / 1024
            new_kb = os.path.getsize(int8_path) / 1024
            console.print(f"[green]  Quantized classifier saved to {int8_path} ({orig_kb:.0f} KB -> {new_kb:.0f} KB)[/green]")
        else:
            console.print("[yellow]  Quantization skipped (onnxruntime not installed).[/yellow]")
