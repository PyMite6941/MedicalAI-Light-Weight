Welcome! This is my attempt on a Medical AI tool that can be run anywhere with great speed. This is my first ever project with VLM, and VLM is necessary for medical procedures so that's why I chose VLM.

## Setup

### 1. Prerequisites
- Python 3.10+
- Git
- Windows (scripts are tested on Windows PowerShell)

### 2. Clone and enter the project

```bash
git clone <repo-url> MedicalAI
cd MedicalAI
```

### 3. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the app

```powershell
python run.py
```

The app starts in interactive mode. From the menu you can:
- **Vision** — pick an X-ray image and generate a radiology caption
- **Symptom Check** — pick an X-ray image, enter symptoms, and get a diagnosis

Optional features (camera capture and DICOM support) install automatically when selected.

### 6. Train the AI models (optional)

To improve accuracy beyond the stock models, train on real medical data:

```powershell
# Download and prepare the training data (IU-Xray + NIH Chest X-ray)
python training.py --mode prepare-data

# Train the fusion model (Symptom Check)
python training.py --mode train --epochs 10 --batch_size 8

# Fine-tune BLIP for radiology caption generation (Vision)
python xray_training.py --mode train --epochs 3 --batch_size 4 --max_samples 500
```

Training flags:

| Flag | What it does | Default |
|---|---|---|
| `--epochs N` | Number of training passes | 5 (training.py) / 3 (xray_training.py) |
| `--batch_size N` | Images per batch | 8 (training.py) / 4 (xray_training.py) |
| `--lr N` | Learning rate | 1e-3 (training.py) / 5e-5 (xray_training.py) |
| `--val_split N` | Fraction held out for validation | 0.15 (training.py) / 0.1 (xray_training.py) |
| `--use_amp` | Mixed precision (faster on GPU) | off |
| `--grad_accum N` | Accumulate gradients over N steps | 1 |
| `--resume` | Resume from last checkpoint (xray_training only) | off |

### 7. Capturing images

When you select **Vision** or **Symptom Check**, you pick how to provide the image:

| Method | What it does | Requires |
|---|---|---|
| Browse files | Opens a native file picker | Nothing |
| Enter path | Type the image path manually | Nothing |
| Capture from camera | Takes a photo from webcam | Installs opencv-python on first use |
| Load DICOM | Opens medical DICOM (.dcm) files | Installs pydicom + numpy on first use |

All captured images are saved to `data/images/` automatically.

## Optimization

Both models are already optimized for low resource usage out of the box:

- **CPU threads** are automatically limited to half your logical cores
- **GPU half-precision (FP16)** activates automatically if a GPU is detected (2x less VRAM)
- **Model caching** keeps the frozen encoders in RAM so they only load once
- **Memory cleanup** unloads models after each inference session

### Quantization

The fusion model's classifier head (Symptom Check) can be exported to ONNX for a minor CPU speedup:

```powershell
python quantization.py --mode quantize-fusion
```

Other useful commands:

```powershell
python quantization.py --mode set-threads    # Limit CPU thread usage
python quantization.py --mode status          # Show current optimization state
python quantization.py --mode explain         # Explain all optimization options
```

> BLIP (Vision model) ONNX export requires a newer version of `optimum`.
> Run `pip install --upgrade optimum` to enable it. The system works fine
> with PyTorch in the meantime -- no missing functionality.
