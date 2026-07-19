import argparse
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

MODEL_DIR = "./blip-xray-finetuned"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "xray_blip.pth")
ONNX_PATH = os.path.join(MODEL_DIR, "onnx")

class RadiologyCaptionDataset(Dataset):
    def __init__(self, hf_dataset, max_samples=None):
        self.data = []
        for i, example in enumerate(hf_dataset):
            if max_samples and i >= max_samples:
                break
            image = example.get("image")
            caption = example.get("caption", "").strip()
            if image is None or not caption:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.data.append((image, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, processor):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    encoding = processor(
        images=images, text=captions, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    encoding["labels"] = encoding["input_ids"].clone()
    return encoding

def train(epochs, batch_size, lr, max_samples, resume, val_split, use_amp, grad_accum):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from datasets import load_dataset

    has_gpu = torch.cuda.is_available()
    use_amp = use_amp and has_gpu
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    hf_ds = load_dataset("eltorio/ROCOv2-radiology", split="train", streaming=True)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    if resume and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
        start_epoch = torch.load(CHECKPOINT_PATH, weights_only=False).get("epoch", 0) + 1
    else:
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        start_epoch = 0

    if has_gpu:
        model = model.cuda()

    from functools import partial
    _collate = partial(collate_fn, processor=processor)

    full_dataset = RadiologyCaptionDataset(hf_ds, max_samples)
    val_size = max(int(val_split * len(full_dataset)), 1)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    print(f"Dataset: {len(train_subset)} train + {len(val_subset)} val samples")
    print(f"Device: {'GPU' if has_gpu else 'CPU'}  AMP: {'ON' if use_amp else 'OFF'}  Grad accum: {grad_accum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*40}")
        print(f"  Epoch {epoch + 1}/{epochs}")
        print(f"{'='*40}")

        # ── Train ──
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"  Train")
        for i, batch in enumerate(pbar):
            pixel_values = batch.get("pixel_values")
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")

            if has_gpu:
                pixel_values = pixel_values.cuda()
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val"):
                pixel_values = batch.get("pixel_values")
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels")
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        model.train()

        avg_val_loss = val_loss / len(val_loader)
        print(f"  Train loss: {avg_train_loss:.4f}  |  Val loss: {avg_val_loss:.4f}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)
        torch.save({"epoch": epoch}, CHECKPOINT_PATH)
        print(f"  Checkpoint saved to {MODEL_DIR}")

    print(f"\nTraining complete. Model saved to {MODEL_DIR}")

def evaluate(batch_size, max_samples):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from datasets import load_dataset

    if not os.path.exists(MODEL_DIR):
        print(f"No model found at {MODEL_DIR}. Run training first.")
        return

    from functools import partial
    print("Loading model and dataset...")
    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()

    _collate = partial(collate_fn, processor=processor)
    hf_ds = load_dataset("eltorio/ROCOv2-radiology", split="train", streaming=True)
    dataset = RadiologyCaptionDataset(hf_ds, max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method4
    except ImportError:
        print("nltk not installed. Skipping BLEU evaluation.")
        print("Install with: pip install nltk")
        return

    print(f"Evaluating on {len(dataset)} samples...")
    references = []
    hypotheses = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pixel_values = batch.get("pixel_values")
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            generated_ids = model.generate(pixel_values=pixel_values, max_length=64)
            for i in range(len(labels)):
                ref = processor.decode(labels[i], skip_special_tokens=True)
                hyp = processor.decode(generated_ids[i], skip_special_tokens=True)
                if ref and hyp:
                    references.append([ref.split()])
                    hypotheses.append(hyp.split())

    avg_loss = total_loss / len(loader)
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Corpus BLEU: {bleu:.4f}")

    print("\nSample generations:")
    for i in range(min(5, len(references))):
        ref = " ".join(references[i][0])
        hyp = " ".join(hypotheses[i])
        print(f"  REF: {ref[:120]}")
        print(f"  HYP: {hyp[:120]}")
        print()

def export_onnx():
    print("[yellow]BLIP ONNX export requires a newer version of optimum.[/yellow]")
    print("[yellow]Run: pip install --upgrade optimum[/yellow]")
    print("[yellow]Until then, the system uses PyTorch directly (no speed difference for inference).[/yellow]")

def generate(image_path):
    from transformers import BlipProcessor, BlipForConditionalGeneration

    if not os.path.exists(MODEL_DIR):
        print(f"No model found at {MODEL_DIR}. Run training first.")
        return

    print(f"Loading model from {MODEL_DIR}...")
    processor = BlipProcessor.from_pretrained(MODEL_DIR)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=64)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Generated caption: {caption}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP for radiology caption generation")
    parser.add_argument("--mode", required=True, choices=["train", "evaluate", "export-onnx", "generate"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples for training/eval (remove to use all)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation")
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision (GPU only)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--image", help="Path to image for --mode generate")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.epochs, args.batch_size, args.lr, args.max_samples, args.resume, args.val_split, args.use_amp, args.grad_accum)
    elif args.mode == "evaluate":
        evaluate(args.batch_size, args.max_samples)
    elif args.mode == "export-onnx":
        export_onnx()
    elif args.mode == "generate":
        if not args.image:
            print("--mode generate requires --image <path>")
        else:
            generate(args.image)
