import os
import sys
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(rank)
    print(f"[DDP] rank={rank} world={world_size} device=cuda:{rank}", flush=True)
    return rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class DiagnosisFusionModel(nn.Module):
    def __init__(self, num_classes, freeze_encoders=True):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.symptom_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.symptom_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.symptom_encoder.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def encode_images(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            return self.image_encoder.get_image_features(**inputs)

    def encode_symptoms(self, symptom_texts):
        inputs = self.symptom_tokenizer(
            symptom_texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        with torch.no_grad():
            outputs = self.symptom_encoder(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    def forward(self, images, symptom_texts):
        image_vecs = self.encode_images(images)
        symptom_vecs = self.encode_symptoms(symptom_texts)
        combined = torch.cat([image_vecs, symptom_vecs], dim=-1)
        return self.classifier(combined)


def train_epoch(model, loader, optimizer, loss_fn, scaler, epoch, rank, use_amp, grad_accum):
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad()

    for i, (images, symptoms, labels) in enumerate(loader):
        labels = labels.cuda(rank, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images, symptoms)
            loss = loss_fn(logits, labels)
        loss = loss / grad_accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum * labels.size(0)
        total_samples += labels.size(0)

        if rank == 0 and i % 50 == 0:
            print(f"  E{epoch} batch {i}/{len(loader)} loss={loss.item() * grad_accum:.4f}", flush=True)

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


def validate(model, loader, loss_fn, rank):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, symptoms, labels in loader:
            labels = labels.cuda(rank, non_blocking=True)
            logits = model(images, symptoms)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    return total_loss / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Mega-Training Entry Point")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--model_dir", type=str, default="/model")
    parser.add_argument("--data_dir", type=str, default="/data")
    parser.add_argument("--gcs_bucket", type=str, default="")
    parser.add_argument("--gcs_data_prefix", type=str, default="data/")
    parser.add_argument("--gcs_model_prefix", type=str, default="models/")
    parser.add_argument("--freeze_encoders", action="store_true", default=True)
    parser.add_argument("--resume_from", type=str, default="")
    args = parser.parse_args()

    use_amp = args.use_amp and torch.cuda.is_available()
    is_ddp = is_distributed()
    rank = 0
    world_size = 1

    if is_ddp:
        rank, world_size = setup_ddp()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Device: {device}  DDP: {is_ddp}  World: {world_size}", flush=True)
        print(f"AMP: {use_amp}  Grad accum: {args.grad_accum}", flush=True)
        print(f"Batch: {args.batch_size}  Epochs: {args.epochs}  LR: {args.lr}", flush=True)

    if args.gcs_bucket and not os.path.exists(args.data_dir):
        if rank == 0:
            print(f"Downloading data from gs://{args.gcs_bucket}/{args.gcs_data_prefix}...", flush=True)
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(args.gcs_bucket)
            os.makedirs(os.path.join(args.data_dir, "images"), exist_ok=True)
            blobs = list(bucket.list_blobs(prefix=args.gcs_data_prefix))
            for blob in blobs:
                if blob.name.endswith(".csv"):
                    blob.download_to_filename(os.path.join(args.data_dir, "dataset.csv"))
                elif blob.name.endswith((".jpg", ".jpeg", ".png")):
                    local = os.path.join(args.data_dir, "images", os.path.basename(blob.name))
                    if not os.path.exists(local):
                        blob.download_to_filename(local)
            print(f"Downloaded {len(blobs)} blobs", flush=True)

    from data_pipeline import MegaFusionDataset, custom_collate, NUM_CLASSES, CANONICAL_LABELS
    csv_path = os.path.join(args.data_dir, "dataset.csv")
    image_dir = os.path.join(args.data_dir, "images")

    dataset = MegaFusionDataset(csv_path, image_dir, max_samples=args.max_samples)
    val_size = max(int(args.val_split * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=custom_collate,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, collate_fn=custom_collate,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    if rank == 0:
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Classes: {NUM_CLASSES}", flush=True)

    model = DiagnosisFusionModel(NUM_CLASSES, freeze_encoders=args.freeze_encoders)
    if args.resume_from and os.path.exists(args.resume_from):
        model.load_state_dict(torch.load(args.resume_from, map_location="cpu", weights_only=False))
        if rank == 0:
            print(f"Resumed from {args.resume_from}", flush=True)

    if is_ddp:
        model = model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.module.classifier.parameters() if is_ddp else model.classifier.parameters(),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch, rank, use_amp, args.grad_accum)
        val_loss = validate(model, val_loader, loss_fn, rank)
        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}", flush=True)

            os.makedirs(args.model_dir, exist_ok=True)
            ckpt_path = os.path.join(args.model_dir, f"fusion_epoch{epoch+1:02d}.pth")
            state = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save({
                "epoch": epoch + 1,
                "model_state": state,
                "classifier_state": model.module.classifier.state_dict() if is_ddp else model.classifier.state_dict(),
                "label_list": CANONICAL_LABELS,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved {ckpt_path}", flush=True)

    if rank == 0:
        final_path = os.path.join(args.model_dir, "fusion_final.pth")
        state = model.module.state_dict() if is_ddp else model.state_dict()
        torch.save({
            "model_state": state,
            "classifier_state": model.module.classifier.state_dict() if is_ddp else model.classifier.state_dict(),
            "label_list": CANONICAL_LABELS,
            "num_classes": NUM_CLASSES,
            "loss_fn": "BCEWithLogitsLoss",
        }, final_path)

        with open(os.path.join(args.model_dir, "labels.json"), "w") as f:
            json.dump(CANONICAL_LABELS, f)
        print(f"Final model saved to {final_path}", flush=True)

        if args.gcs_bucket:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(args.gcs_bucket)
            for fname in os.listdir(args.model_dir):
                local = os.path.join(args.model_dir, fname)
                if os.path.isfile(local):
                    blob = bucket.blob(f"{args.gcs_model_prefix}{fname}")
                    blob.upload_from_filename(local)
                    print(f"Uploaded {fname} to gs://{args.gcs_bucket}/{args.gcs_model_prefix}{fname}", flush=True)

    if is_ddp:
        cleanup_ddp()

    if rank == 0:
        print("Training complete.", flush=True)


if __name__ == "__main__":
    main()