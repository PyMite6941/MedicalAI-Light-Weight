"""
Inference for the bone X-ray fracture screening model (see
bone_xray_training.py). Kept as a separate, small module — this is a
distinct model/domain from the chest X-ray pipeline in optimize.py, with
its own checkpoint and label space, so it shouldn't be tangled into that
file. It reuses optimize.py's device detection and memory-clearing so
device/resource behavior stays consistent across the whole app.
"""
import os

from PIL import Image

from bone_xray_training import BONE_LABELS, CHECKPOINT_PATH, BoneFractureModel
from optimize import get_torch_device, get_device

_loaded_model = None


def model_available():
    return os.path.exists(CHECKPOINT_PATH)


def infer_bone_fracture(image_path, threshold=0.5):
    """Returns (findings, confidences) where `findings` is a list of
    labels (from BONE_LABELS) predicted above `threshold`, and
    `confidences` maps every label to its raw probability.

    Returns (None, message) if no trained checkpoint exists yet — this is
    a scaffold path, not something that ships pretrained.
    """
    global _loaded_model
    import torch

    if not model_available():
        return None, (
            f"No bone fracture model trained yet. Run:\n"
            f"  python bone_xray_training.py --mode prepare-data\n"
            f"  python bone_xray_training.py --mode train"
        )

    if _loaded_model is None:
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
        labels = checkpoint.get("labels", BONE_LABELS)
        model = BoneFractureModel(num_labels=len(labels))
        model.classifier.load_state_dict(checkpoint["model_state"])
        model.eval()
        if get_device() not in ("tpu", "cpu"):
            model = model.to(get_torch_device())
        _loaded_model = {"model": model, "labels": labels}

    model = _loaded_model["model"]
    labels = _loaded_model["labels"]

    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        logits = model([image])
        probs = torch.sigmoid(logits)[0]

    confidences = {labels[i]: probs[i].item() for i in range(len(labels))}
    findings = [label for label, p in confidences.items() if p >= threshold]
    return findings, confidences


def clear_bone_model_cache():
    global _loaded_model
    _loaded_model = None
