"""
Download pre-trained checkpoints so users can skip training from scratch.

Usage:
  python download_model.py                     # interactive menu
  python download_model.py --all               # download everything
  python download_model.py --fusion            # fusion model only
  python download_model.py --blip-finetuned    # fine-tuned BLIP only
"""
import argparse
import os
import sys
from pathlib import Path

BASE_URL = "https://huggingface.co/your-org/medicalai-lightweight/resolve/main"
CHECKPOINT_DIR = Path("./checkpoints")
BLIP_DIR = Path("./blip-xray-finetuned")
ONNX_DIR = CHECKPOINT_DIR / "onnx_full"


def _ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url, dest):
    import requests
    from rich.console import Console
    from rich.progress import Progress, BarColumn, DownloadColumn, TextColumn

    console = Console()
    console.print(f"[cyan]Downloading {url.split('/')[-1]}...[/cyan]")

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with Progress(
        TextColumn("[cyan]  Download[/cyan]"),
        BarColumn(),
        DownloadColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("", total=total)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(task, advance=len(chunk))

    console.print(f"[green]  Saved to {dest}[/green]")


def download_fusion():
    _ensure_dir(CHECKPOINT_DIR)
    _ensure_dir(ONNX_DIR)

    files = [
        ("fusion_model.pth", CHECKPOINT_DIR / "fusion_model.pth"),
        ("fusion_full.onnx", ONNX_DIR / "fusion_full.onnx"),
        ("labels.json", ONNX_DIR / "labels.json"),
    ]

    for fname, dest in files:
        url = f"{BASE_URL}/{fname}"
        print(f"  Would download: {url} -> {dest}")

    print()
    print("[yellow]Note: Pre-trained checkpoints are not yet hosted.[/yellow]")
    print("[yellow]Train locally with: python training.py --mode prepare-data && python training.py --mode train[/yellow]")
    print("[yellow]Then export:       python quantization.py --mode export-full[/yellow]")


def download_blip():
    _ensure_dir(BLIP_DIR)

    files = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ]

    for fname in files:
        url = f"{BASE_URL}/blip-xray-finetuned/{fname}"
        dest = BLIP_DIR / fname
        print(f"  Would download: {url} -> {dest}")

    print()
    print("[yellow]Note: Fine-tuned BLIP is not yet hosted.[/yellow]")
    print("[yellow]Train locally with: python xray_training.py --mode train[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--fusion", action="store_true", help="Download fusion model")
    parser.add_argument("--blip-finetuned", action="store_true", help="Download fine-tuned BLIP")
    args = parser.parse_args()

    if not any([args.all, args.fusion, args.blip_finetuned]):
        from rich.console import Console
        import questionary
        console = Console()
        console.print("[bold cyan]Download Pre-trained Models[/bold cyan]")
        choice = questionary.select(
            "What would you like to download?",
            choices=[
                "Fusion model (Symptom Check) — ONNX + PyTorch",
                "Fine-tuned BLIP (Vision captioning)",
                "Everything",
                "Cancel",
            ],
        ).ask()
        if choice == "Cancel":
            return
        if choice == "Fusion model (Symptom Check) — ONNX + PyTorch":
            args.fusion = True
        elif choice == "Fine-tuned BLIP (Vision captioning)":
            args.blip_finetuned = True
        else:
            args.all = True

    try:
        import requests
    except ImportError:
        print("'requests' is required. Install with: pip install requests")
        sys.exit(1)

    if args.all or args.fusion:
        download_fusion()
    if args.all or args.blip_finetuned:
        download_blip()


if __name__ == "__main__":
    main()
