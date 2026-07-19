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

def infer_fusion(image_path, symptoms):
    global _loaded_fusion
    if _loaded_fusion is None:
        if not os.path.exists(CHECKPOINT_PATH):
            return None, "No trained fusion model found"
        from training import DiagnosisFusionModel, load_label_list
        label_list = load_label_list()
        if not label_list:
            return None, "No label data found"
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        model = DiagnosisFusionModel(num_conditions=len(label_list))
        model.classifier.load_state_dict(checkpoint["model_state"])
        _loaded_fusion = {
            "model": model.eval(),
            "label_list": label_list,
        }

    image = Image.open(image_path).convert("RGB")
    m = _loaded_fusion["model"]
    label_list = _loaded_fusion["label_list"]

    from training import collate_fn
    with torch.no_grad():
        logits = m([image], [symptoms])
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)

    return label_list[predicted.item()], confidence.item()

# ── Quantization ──────────────────────────────────────────────

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
