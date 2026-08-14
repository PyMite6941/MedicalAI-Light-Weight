Medical AI - Light Weight: chest X-ray analysis tool that runs on consumer hardware.

## Quick Start

**Windows:** Double-click `launch.ps1` — it handles everything (Python check, venv, install, default model, launch).

**Linux/Mac:** `bash launch.sh`

**Manual:**
```powershell
pip install -r requirements.txt
python setup_default.py    # generates default ONNX model (5 KB, random weights)
python web_ui.py           # browser at localhost:7860
```

The app works immediately with default models. Results improve after training.

## Usage

### Browser UI (easiest for non-technical users)

```powershell
pip install gradio   # first time only
python web_ui.py
# → Opens http://127.0.0.1:7860
```

Upload an X-ray and optionally enter symptoms. Two tabs:
- **Vision** — generates a radiology caption/report from the image
- **Symptom Check** — enter symptoms + image → diagnosis with confidence score, plus a plain-language
  explanation of the finding, its typical urgency, and a suggested next step (see `knowledge.py`).
  This is general education content, not a diagnosis — always confirm with a clinician.

### CLI (interactive)

```powershell
python run.py
```

Menu-driven: pick an image, run Vision or Symptom Check.

### Batch (process many images at once)

```powershell
python batch_predict.py ./data/images/ -o results.csv --symptoms "Cough and fever" --vision
python batch_predict.py single_image.jpg --onnx
```

Output is a CSV with columns: image, symptoms, caption, diagnosis, confidence.

## Training

```powershell
# 1. Download IU-Xray + NIH Chest X-ray datasets (~10 GB)
python training.py --mode prepare-data

# 2. Train the fusion model (Symptom Check)
python training.py --mode train --epochs 10 --batch_size 8

# 3. (Optional) Expand dataset with 160 rare diagnosis entries
python expand_dataset.py

# 4. (Optional) Fine-tune BLIP for better Vision captions
python xray_training.py --mode train --epochs 3 --batch_size 4 --max_samples 500
```

## Deployment Options

### Option A: PyTorch (default)
Works immediately after `pip install -r requirements.txt`. Full flexibility.

### Option B: ONNX (no PyTorch at inference)

```powershell
# Export the entire image+symptom→diagnosis pipeline to ONNX
python quantization.py --mode export-full
```

This creates `./checkpoints/onnx_full/fusion_full.onnx` — a standalone model that runs with just `onnxruntime`:
- No `torch` needed at inference time
- ~50% faster on CPU
- ~60% less memory

Every export is automatically followed by **INT8 dynamic quantization**, producing a
`fusion_full.int8.onnx` alongside it (roughly a further 4x smaller, faster on CPU).
Inference always prefers the quantized file when one exists — nothing extra to run.

Run with: `python batch_predict.py image.jpg --onnx` or check "Use ONNX" in the web UI.

### Option C: Standalone .exe (no Python needed on target machine)

```powershell
pip install pyinstaller
python build_exe.py          # builds web_ui.exe
python build_exe.py --cli    # builds run.exe (CLI)
python build_exe.py --all    # builds both + batch_predict.exe
```

Copy the `dist/` folder + `checkpoints/` to any Windows machine — no Python install needed.

### Option D: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && pip install gradio
CMD ["python", "web_ui.py"]
```

Build & run: `docker build -t medicalai . && docker run -p 7860:7860 medicalai`

## Optimization

```powershell
python quantization.py --mode export-full       # Full ONNX pipeline (+ INT8 quantization)
python quantization.py --mode quantize-fusion    # Classifier head only (+ INT8 quantization)
python quantization.py --mode quantize-blip      # Export + quantize the BLIP vision model to ONNX
python quantization.py --mode quantize-int8      # INT8-quantize any ONNX models that exist but aren't yet
python quantization.py --mode optimize-all       # Everything above, end to end
python quantization.py --mode set-threads        # Limit CPU usage
python quantization.py --mode status             # Current state (device, providers, quantized models)
python quantization.py --mode explain            # Detailed explanation
```

The system already auto-tunes:
- **Device**: auto-detects in priority order — TPU (via `torch_xla`, if installed and a TPU runtime
  is present) → CUDA (NVIDIA GPU) → MPS (Apple Silicon GPU) → CPU. Same priority applies to ONNX
  Runtime execution providers.
- **Quantization**: every ONNX export (fusion classifier, full pipeline, BLIP vision model) is
  automatically followed by INT8 dynamic quantization. Inference always prefers the quantized
  (`*.int8.onnx`) file when one is present.
- CPU threads limited to half the logical cores, applied consistently to PyTorch, ONNX Runtime,
  and MKL/OpenMP — training scripts included
- GPU/MPS FP16 auto-detected
- Model, ONNX session, and preprocessor caching (load once, reuse — no more re-loading tokenizers
  or rebuilding inference sessions on every request)
- Memory cleanup after each inference

## Default Models

`python setup_default.py` generates `models/default/fusion_classifier.onnx` — a 1.3 MB ONNX model with 15 standard NIH chest X-ray classes and random weights. This lets the app run immediately after install:

- **Symptom Check** runs with default ONNX model + PyTorch encoders (CLIP + BERT)
- **Vision** uses stock BLIP from Hugging Face (always works)
- Replace with a trained model for real accuracy

## Updating

```powershell
# Interactive menu
python update.py

# Download latest models from Hugging Face (set your repo first)
python update.py --set-url your-org/your-models
python update.py --models

# Pull latest code from git
python update.py --code

# Upgrade pip packages
python update.py --deps

# Everything at once
python update.py --all
```

Config is saved in `update_config.json`.

## Project Structure

| File | Purpose |
|------|---------|
| `web_ui.py` | **Browser UI** — `python web_ui.py`, opens at localhost:7860 |
| `run.py` | Interactive CLI |
| `training.py` | Download data + train fusion model |
| `optimize.py` | Device detection, inference, memory, ONNX export/quantization, CPU control |
| `quantization.py` | CLI wrapper for optimization commands + REST API server |
| `knowledge.py` | Per-condition descriptions, symptoms, urgency, and follow-up guidance |
| `expand_dataset.py` | Downloads large public datasets + ~200 rare/obscure diagnosis entries |
| `batch_predict.py` | Non-interactive batch prediction → CSV |
| `capture.py` | Image input: file picker, camera, DICOM |
| `download_model.py` | Download pre-trained checkpoints (future) |
| `build_exe.py` | Build standalone .exe with PyInstaller |
| `launch.ps1` | One-click Windows launcher |
| `launch.sh` | One-click Linux/Mac launcher |
| `config.json` | Settings file (edit instead of Python code) |

## Dataset

| Source | Rows |
|--------|------|
| IU-Xray | 6,687 |
| NIH Chest X-ray | 3,000 |
| Augmented (rare findings) | ~200 |
| **Total** | **~9,887** |

Covers 50+ finding combinations: normal, cardiomegaly, CHF, pneumonia (lobar/round/cavitary/viral), atelectasis, pleural effusion, pneumothorax, COPD/emphysema, nodules/masses, ILD/fibrosis, TB, bronchiectasis, fractures, hiatal hernia, pneumoperitoneum, aortic aneurysm/dissection, pericardial effusion, PE signs, congenital anomalies, pneumoconiosis (silicosis/asbestosis/CWP), LAM, LCH, alveolar proteinosis, Swyer-James, ABPA, scimitar syndrome, and more.

## Requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended)
- ~3 GB free disk for model downloads
- Runs fully on CPU; TPU, NVIDIA GPU (CUDA), and Apple Silicon GPU (MPS) are all auto-detected and
  used automatically when available, in that priority order — no configuration needed

## Lower-tech / Small Company Guide

This tool is designed for a single laptop or workstation. No cloud, no GPU farm, no IT team.

### Best setup for a non-technical clinic

1. Install Python 3.10+ (check "Add to PATH" during install)
2. Double-click `launch.ps1` — it installs everything, trains the model, and opens the browser
3. Bookmark `http://127.0.0.1:7860`
4. That's it — upload X-rays, type symptoms, get results

### To avoid training entirely (no dataset download)

Use the ONNX export path:
1. Get a pre-trained `fusion_full.onnx` + `labels.json` from a colleague who already trained
2. Place them in `./checkpoints/onnx_full/`
3. Run `python web_ui.py` — check "Use ONNX" in the web UI
4. No PyTorch, no training, no dataset download needed

### To distribute to multiple computers without Python

1. Run `python build_exe.py --all` on the machine that has everything installed
2. Copy `dist/web_ui.exe` + `checkpoints/` folder to target machines
3. Double-click the .exe — no Python install needed

### Tips for older / low-spec hardware

- `python quantization.py --mode set-threads` — limits CPU usage
- Use ONNX mode — ~50% faster, less memory
- Disable Vision (BLIP) if not needed — it's the heavier model
- Run on a laptop with 8 GB RAM works fine for single-user use
