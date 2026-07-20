"""
Batch prediction script for non-interactive use.
Processes many X-ray images (or a single one) and outputs results as CSV.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

from optimize import (
    clear_memory,
    infer_blip,
    infer_fusion,
    infer_fusion_onnx,
    set_cpu_threads,
)

CHECKPOINT_PATH = "./checkpoints/fusion_model.pth"
ONNX_FULL_DIR = "./checkpoints/onnx_full"
HAS_FUSION = os.path.exists(CHECKPOINT_PATH)
HAS_ONNX = os.path.exists(os.path.join(ONNX_FULL_DIR, "fusion_full.onnx"))


def find_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"}
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in exts])


def process_image(image_path, symptoms, use_vision, use_onnx):
    result = {"image": str(image_path), "symptoms": symptoms}

    if use_vision:
        try:
            caption = infer_blip(str(image_path), use_onnx=HAS_ONNX)
            result["caption"] = caption
        except Exception as e:
            result["caption"] = f"ERROR: {e}"

    if HAS_FUSION and symptoms:
        try:
            if use_onnx and HAS_ONNX:
                diagnosis, confidence = infer_fusion_onnx(str(image_path), symptoms)
            else:
                diagnosis, confidence = infer_fusion(str(image_path), symptoms)
            result["diagnosis"] = diagnosis if diagnosis else "INCONCLUSIVE"
            result["confidence"] = f"{confidence:.4f}" if isinstance(confidence, float) else confidence
        except Exception as e:
            result["diagnosis"] = f"ERROR: {e}"
            result["confidence"] = ""

    clear_memory()
    return result


def main():
    set_cpu_threads()

    parser = argparse.ArgumentParser(description="Batch X-ray prediction")
    parser.add_argument("input", help="Path to image file or directory")
    parser.add_argument("--symptoms", default="What abnormality is present in this chest X-ray?",
                        help="Symptoms text (same for all images)")
    parser.add_argument("--output", "-o", default="predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--vision", action="store_true",
                        help="Run vision captioning (BLIP)")
    parser.add_argument("--onnx", action="store_true",
                        help="Use ONNX runtime (if exported)")
    args = parser.parse_args()

    images = find_images(args.input)
    if not images:
        print(f"No images found at: {args.input}")
        sys.exit(1)

    print(f"Found {len(images)} image(s)")
    print(f"Symptoms: {args.symptoms}")
    print(f"Vision: {'ON' if args.vision else 'OFF'}")
    print(f"ONNX: {'ON' if args.onnx and HAS_ONNX else 'OFF'}")
    print()

    results = []
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {img_path.name}...")
        result = process_image(str(img_path), args.symptoms, args.vision, args.onnx)
        results.append(result)

    fieldnames = ["image", "symptoms", "caption", "diagnosis", "confidence"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
