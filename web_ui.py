"""
Browser-based UI for MedicalAI - Light Weight.
Auto-installs gradio if missing.
Auto-generates default models if none are trained.
"""
import importlib
import os
import subprocess
import sys
import tempfile

from PIL import Image

from bone_fracture import infer_bone_fracture, model_available as bone_model_available
from drug_interactions import LEVEL_FROM_STR, check_medication_list, db_available
from drug_interactions import DISCLAIMER as DRUG_DISCLAIMER
from knowledge import get_condition_info, DISCLAIMER
from optimize import (
    clear_memory,
    get_available_models,
    get_device,
    infer_blip,
    infer_fusion,
    infer_fusion_onnx,
    set_cpu_threads,
)

CHECKPOINT_PATH = "./checkpoints/fusion_model.pth"
ONNX_FULL_DIR = "./checkpoints/onnx_full"
DEFAULT_MODEL_DIR = "./models/default"


def _ensure_gradio():
    try:
        return importlib.import_module("gradio")
    except ImportError:
        from rich.console import Console
        console = Console()
        console.print("[yellow]'gradio' is required for the web UI.[/yellow]")
        import questionary
        install = questionary.confirm("Install gradio now?", default=True).ask()
        if not install:
            console.print("[red]gradio is required. Exiting.[/red]")
            sys.exit(1)
        console.print("[cyan]Installing gradio...[/cyan]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        return importlib.import_module("gradio")


def _ensure_default_models():
    """Generate default ONNX models if none exist at all."""
    if os.path.exists(os.path.join(DEFAULT_MODEL_DIR, "fusion_classifier.onnx")):
        return
    if os.path.exists(CHECKPOINT_PATH):
        return
    print("No models found. Generating default models...")
    subprocess.check_call(
        [sys.executable, "setup_default.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("Default models ready.")


def analyze_vision(image):
    if image is None:
        return "Please upload an X-ray image."
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        Image.fromarray(image).save(path)
    try:
        caption = infer_blip(path, use_onnx=False)
        return caption
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.unlink(path)
        clear_memory()


def analyze_symptom(image, symptoms, use_onnx):
    if image is None:
        return "Please upload an X-ray image.", "", ""
    if not symptoms:
        symptoms = "No symptoms provided"
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        Image.fromarray(image).save(path)
    try:
        if use_onnx:
            diagnosis, confidence = infer_fusion_onnx(path, symptoms)
        else:
            diagnosis, confidence = infer_fusion(path, symptoms)
        if diagnosis is None:
            return confidence, "No result", ""

        info = get_condition_info(diagnosis)
        context_lines = [info["description"]]
        if info["common_symptoms"]:
            context_lines.append("Commonly associated with: " + ", ".join(info["common_symptoms"]) + ".")
        context_lines.append(f"Urgency: {info['urgency']}. {info['follow_up']}")
        context_lines.append(DISCLAIMER)

        return diagnosis, f"{confidence:.1%}", "\n".join(context_lines)
    except Exception as e:
        return f"Error: {e}", "", ""
    finally:
        os.unlink(path)
        clear_memory()


def analyze_medications(meds_text):
    if not meds_text or not meds_text.strip():
        return "Enter medications, one per line or comma-separated."

    meds = [m.strip() for line in meds_text.splitlines() for m in line.split(",") if m.strip()]
    if len(meds) < 2:
        return "Enter at least 2 medications to check for interactions."

    interactions, unrecognized = check_medication_list(meds)

    lines = []
    if interactions:
        for r in sorted(interactions, key=lambda x: LEVEL_FROM_STR.get(x["level"], 0), reverse=True):
            line = f"{r['drug_a']} + {r['drug_b']}: {r['level']}"
            if r.get("note"):
                line += f" — {r['note']}"
            line += f" (source: {r['source']})"
            lines.append(line)
    else:
        lines.append("No known interactions found among these medications.")

    if unrecognized:
        lines.append(f"Not found in the interaction database: {', '.join(unrecognized)}")
    if not db_available():
        lines.append("Using the small built-in reference set only. Run 'python drug_interactions.py --download' for much broader coverage (needs network, one-time).")
    lines.append("")
    lines.append(DRUG_DISCLAIMER)

    return "\n".join(lines)


def analyze_bone(image):
    if image is None:
        return "Please upload an X-ray image."
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        Image.fromarray(image).save(path)
    try:
        findings, confidences = infer_bone_fracture(path)
        if findings is None:
            return confidences  # "not trained yet" message

        lines = []
        if findings:
            for label in findings:
                info = get_condition_info(label)
                lines.append(f"{label} ({confidences[label]:.1%}): {info['description']}")
        else:
            lines.append("No fracture or hardware detected above threshold.")
        lines.append("")
        lines.append("All confidences: " + ", ".join(f"{k}={v:.1%}" for k, v in confidences.items()))
        lines.append(DISCLAIMER)
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.unlink(path)
        clear_memory()


def main():
    set_cpu_threads()
    _ensure_default_models()
    gr = _ensure_gradio()

    models = get_available_models()
    has_trained = models["trained_pytorch"]
    has_default = models["default_classifier"]
    has_onnx_full = models["onnx_full_pipeline"]

    model_source = "ONNX (full pipeline)" if has_onnx_full else \
                   "Trained PyTorch" if has_trained else \
                   "Default (random weights)" if has_default else \
                   "NOT AVAILABLE"

    print(f"Device: {get_device().upper()}")
    print(f"Model:  {model_source}")
    print()
    print("Launching web UI... open the URL below in your browser.")

    with gr.Blocks(title="MedicalAI - Light Weight", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # MedicalAI - Light Weight
            Upload a chest X-ray for AI-assisted analysis.
            """
        )

        with gr.Tab("Vision - Generate Report"):
            gr.Markdown("Upload an X-ray and get an AI-generated radiology caption.")
            with gr.Row():
                img_in = gr.Image(label="X-ray Image", type="numpy")
            with gr.Row():
                btn_vision = gr.Button("Generate Report", variant="primary")
            with gr.Row():
                caption_out = gr.Textbox(label="Radiology Caption", lines=6)

            btn_vision.click(fn=analyze_vision, inputs=img_in, outputs=caption_out)

        with gr.Tab("Symptom Check - Diagnosis"):
            gr.Markdown("Upload an X-ray, enter symptoms, and get a diagnosis.")
            with gr.Row():
                img_in2 = gr.Image(label="X-ray Image", type="numpy")
            with gr.Row():
                symptoms_in = gr.Textbox(
                    label="Patient Symptoms / Clinical Indication",
                    placeholder="e.g., shortness of breath, cough, fever...",
                    lines=3,
                )
            with gr.Row():
                onnx_checkbox = gr.Checkbox(
                    label="Use ONNX (no PyTorch backend)",
                    value=has_onnx_full,
                    interactive=has_onnx_full,
                )
            with gr.Row():
                btn_diag = gr.Button("Diagnose", variant="primary")
            with gr.Row():
                with gr.Column():
                    diag_out = gr.Textbox(label="Diagnosis", lines=4)
                    conf_out = gr.Textbox(label="Confidence")
            with gr.Row():
                context_out = gr.Textbox(label="Clinical Context (general education, not a diagnosis)", lines=5)

            btn_diag.click(
                fn=analyze_symptom,
                inputs=[img_in2, symptoms_in, onnx_checkbox],
                outputs=[diag_out, conf_out, context_out],
            )

        with gr.Tab("Bone X-ray (experimental)"):
            gr.Markdown(
                "Upload an extremity X-ray (hand/leg/hip/shoulder) to screen for fracture or surgical hardware. "
                "**Separate, experimental model** — " +
                ("a trained checkpoint is available." if bone_model_available() else
                 "no trained checkpoint yet. Run `python bone_xray_training.py --mode prepare-data` "
                 "then `--mode train` to enable this tab.")
            )
            with gr.Row():
                bone_img_in = gr.Image(label="Extremity X-ray", type="numpy")
            with gr.Row():
                btn_bone = gr.Button("Screen for Fracture", variant="primary")
            with gr.Row():
                bone_out = gr.Textbox(label="Result (general education, not a diagnosis)", lines=6)

            btn_bone.click(fn=analyze_bone, inputs=bone_img_in, outputs=bone_out)

        with gr.Tab("Medication Check"):
            gr.Markdown(
                "Enter the patient's current medications (one per line, or comma-separated) "
                "to screen for known interactions. " +
                ("Using the full offline database." if db_available() else
                 "Using the small built-in reference set — run `python drug_interactions.py --download` "
                 "once while online for much broader coverage.")
            )
            with gr.Row():
                meds_in = gr.Textbox(
                    label="Current Medications",
                    placeholder="e.g.\nWarfarin\nIbuprofen\nMetoprolol",
                    lines=5,
                )
            with gr.Row():
                btn_meds = gr.Button("Check Interactions", variant="primary")
            with gr.Row():
                meds_out = gr.Textbox(label="Interaction Results (general education, not a diagnosis)", lines=8)

            btn_meds.click(fn=analyze_medications, inputs=meds_in, outputs=meds_out)

        with gr.Tab("System Info"):
            gr.Markdown(f"""
            **Device:** `{get_device().upper()}`
            **Model:** `{model_source}`
            **Trained checkpoint:** {'Yes' if has_trained else 'No'}
            **ONNX full pipeline:** {'Yes' if has_onnx_full else 'No'}
            **Default model:** {'Yes' if has_default else 'No'}
            **Dataset:** `{os.path.abspath('./data/dataset.csv')}`

            ### 3 ways to improve accuracy
            1. **Train** — `python training.py --mode prepare-data && python training.py --mode train`
            2. **Update** — `python update.py --models` (downloads pre-trained model from Hugging Face)
            3. **Export to ONNX** — `python quantization.py --mode export-full`
            """)

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
