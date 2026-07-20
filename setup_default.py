"""
Generate default ONNX models so the app works immediately after install.

Run: python setup_default.py

Creates:
  - models/default/labels.json          (15 standard NIH classes)
  - models/default/fusion_classifier.onnx (small MLP with random weights)
  - models/default/fusion_full.onnx      (if full pipeline export possible)
"""
import json
import os
import shutil

DEFAULT_DIR = os.path.join("models", "default")
NIH_LABELS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _export_classifier_onnx():
    """Export a minimal ONNX classifier head (random weights, correct shape)."""
    import torch
    import torch.nn as nn

    classifier = nn.Sequential(
        nn.Linear(512 + 768, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(NIH_LABELS)),
    )
    classifier.eval()

    dummy = torch.randn(1, 512 + 768)
    onnx_path = os.path.join(DEFAULT_DIR, "fusion_classifier.onnx")
    torch.onnx.export(
        classifier,
        dummy,
        onnx_path,
        input_names=["features"],
        output_names=["logits"],
        opset_version=14,
    )
    return onnx_path


def _export_full_onnx():
    """Try to export a full-pipeline ONNX using stock CLIP + BERT encoders.

    This requires transformers to be installed. The full model is large
    (~1 GB) but runs standalone with ONNX Runtime (no torch).
    """
    try:
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, AutoModel

        class _MinimalPipeline(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                self.vision_encoder = clip.vision_model
                self.visual_projection = clip.visual_projection
                self.text_encoder = bert
                self.text_pooler = bert.pooler
                self.classifier = nn.Sequential(
                    nn.Linear(512 + 768, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_classes),
                )
                self.classifier.eval()

            def forward(self, pixel_values, input_ids, attention_mask):
                v_out = self.vision_encoder(pixel_values)
                v_feat = self.visual_projection(v_out.pooler_output)
                v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

                t_out = self.text_encoder(input_ids, attention_mask=attention_mask)
                t_feat = self.text_pooler(t_out.last_hidden_state[:, 0, :])

                combined = torch.cat([v_feat, t_feat], dim=-1)
                return self.classifier(combined)

        model = _MinimalPipeline(len(NIH_LABELS)).eval()
        dummy_pixel = torch.randn(1, 3, 224, 224)
        dummy_ids = torch.randint(0, 100, (1, 64), dtype=torch.long)
        dummy_mask = torch.ones(1, 64, dtype=torch.long)

        onnx_path = os.path.join(DEFAULT_DIR, "fusion_full.onnx")
        torch.onnx.export(
            model,
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
        )
        return onnx_path
    except Exception as e:
        return None


def _save_labels():
    path = os.path.join(DEFAULT_DIR, "labels.json")
    with open(path, "w") as f:
        json.dump(NIH_LABELS, f)
    return path


def _save_symptoms():
    """Save default symptom prompts for the Vision model."""
    path = os.path.join(DEFAULT_DIR, "symptoms.txt")
    templates = [
        "What abnormality is present in this chest X-ray?",
        "Patient presents with shortness of breath and cough.",
        "Fever and productive cough for 5 days.",
        "Chest pain and dyspnea on exertion.",
        "Routine pre-operative chest X-ray.",
        "Trauma patient. Evaluate for pneumothorax or fractures.",
        "Patient with history of smoking. Evaluate for lung pathology.",
        "Immunocompromised patient with fever.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(templates))
    return path


def main():
    from rich.console import Console
    console = Console()

    _ensure_dir(DEFAULT_DIR)

    # Remove old default files
    for f in os.listdir(DEFAULT_DIR):
        fp = os.path.join(DEFAULT_DIR, f)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except Exception:
            pass

    console.print("[bold cyan]Generating default models...[/bold cyan]")

    labels_path = _save_labels()
    console.print(f"  [green]{labels_path}[/green]")

    symp_path = _save_symptoms()
    console.print(f"  [green]{symp_path}[/green]")

    cls_path = _export_classifier_onnx()
    size = os.path.getsize(cls_path) / 1024
    console.print(f"  [green]{cls_path}[/green] ({size:.0f} KB)")

    full_path = _export_full_onnx()
    if full_path:
        size = os.path.getsize(full_path) / 1024 / 1024
        console.print(f"  [green]{full_path}[/green] ({size:.0f} MB)")
    else:
        console.print("  [yellow]Full pipeline ONNX export skipped (no transformers or CUDA)[/yellow]")
        console.print("  [yellow]  Run 'python quantization.py --mode export-full' after training for this.[/yellow]")

    console.print()
    console.print("[green]Default models ready. The app will use these until you train a proper model.[/green]")


if __name__ == "__main__":
    main()
