"""
Update script — pulls the newest models, UI files, and dependencies.

Usage:
  python update.py              # interactive menu
  python update.py --all        # update everything
  python update.py --models     # download latest ONNX models
  python update.py --code       # git pull latest source
  python update.py --deps       # upgrade pip packages

The default model source is a Hugging Face repo.
Configure with: python update.py --set-url <HF_REPO_ID>
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG_FILE = "update_config.json"
DEFAULT_HF_REPO = "your-org/medicalai-models"
DEFAULT_DIR = Path("models") / "default"
ONNX_FULL_DIR = Path("checkpoints") / "onnx_full"


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"hf_repo": DEFAULT_HF_REPO, "auto_update": True}


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _file_size(path):
    return os.path.getsize(path) / 1024 / 1024


def update_models_from_hf():
    """Download latest ONNX models from Hugging Face."""
    cfg = load_config()
    repo = cfg["hf_repo"]

    from rich.console import Console
    console = Console()

    if repo == DEFAULT_HF_REPO and "your-org" in repo:
        console.print("[yellow]HF_REPO not set. Skipping model download.[/yellow]")
        console.print("[yellow]Set your model repo: python update.py --set-url your-org/your-repo[/yellow]")
        console.print("[yellow]Or train locally: python training.py --mode prepare-data && python training.py --mode train[/yellow]")
        return

    try:
        import requests
    except ImportError:
        console.print("[red]'requests' required. Install: pip install requests[/red]")
        return

    _ensure_dir(ONNX_FULL_DIR)
    _ensure_dir(DEFAULT_DIR)

    files_to_download = [
        ("fusion_full.onnx", ONNX_FULL_DIR / "fusion_full.onnx"),
        ("labels.json", ONNX_FULL_DIR / "labels.json"),
        ("fusion_classifier.onnx", DEFAULT_DIR / "fusion_classifier.onnx"),
    ]

    base_url = f"https://huggingface.co/{repo}/resolve/main"

    for fname, dest in files_to_download:
        url = f"{base_url}/{fname}"
        console.print(f"[cyan]Downloading {fname}...[/cyan]")
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            console.print(f"  [green]Saved {dest} ({_file_size(dest):.1f} MB)[/green]")
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")

    console.print("[green]Model update complete.[/green]")


def update_code():
    """Pull latest source code from git."""
    from rich.console import Console
    console = Console()

    if not os.path.exists(".git"):
        console.print("[yellow]Not a git repository. Skipping code update.[/yellow]")
        return

    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            console.print(f"[green]{result.stdout}[/green]")
        else:
            console.print(f"[yellow]{result.stderr}[/yellow]")
    except Exception as e:
        console.print(f"[red]Git pull failed: {e}[/red]")


def update_deps():
    """Upgrade all pip packages to latest compatible versions."""
    from rich.console import Console
    console = Console()

    req = "requirements.txt"
    if not os.path.exists(req):
        console.print("[yellow]No requirements.txt found.[/yellow]")
        return

    console.print("[cyan]Upgrading dependencies...[/cyan]")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "-r", req],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        console.print("[green]Dependencies upgraded.[/green]")
    except Exception as e:
        console.print(f"[red]Upgrade failed: {e}[/red]")


def update_all():
    from rich.console import Console
    console = Console()

    console.print("[bold cyan]Full Update[/bold cyan]")
    console.print()

    console.print("[cyan]Step 1: Updating code...[/cyan]")
    update_code()

    console.print("[cyan]Step 2: Upgrading dependencies...[/cyan]")
    update_deps()

    console.print("[cyan]Step 3: Downloading latest models...[/cyan]")
    update_models_from_hf()

    console.print()
    console.print("[green]Update complete![/green]")
    console.print("  Run [bold]python quantization.py --mode status[/bold] to verify.")


def set_repo_url(url):
    cfg = load_config()
    cfg["hf_repo"] = url
    save_config(cfg)
    print(f"Hugging Face repo set to: {url}")


def show_status():
    cfg = load_config()
    print(f"Update config: {CONFIG_FILE}")
    print(f"  HF repo:     {cfg['hf_repo']}")
    print(f"  Auto update: {cfg['auto_update']}")
    print()
    print("Default models:")
    for f in ["fusion_classifier.onnx", "labels.json"]:
        p = DEFAULT_DIR / f
        exists = os.path.exists(p)
        size = f"({_file_size(p):.1f} MB)" if exists else ""
        print(f"  {f}: {'yes' if exists else 'no'} {size}")
    print()
    print("Full ONNX pipeline:")
    for f in ["fusion_full.onnx", "labels.json"]:
        p = ONNX_FULL_DIR / f
        exists = os.path.exists(p)
        size = f"({_file_size(p):.1f} MB)" if exists else ""
        print(f"  {f}: {'yes' if exists else 'no'} {size}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Update MedicalAI models, code, and deps")
    parser.add_argument("--models", action="store_true", help="Download latest ONNX models")
    parser.add_argument("--code", action="store_true", help="Git pull latest source")
    parser.add_argument("--deps", action="store_true", help="Upgrade pip packages")
    parser.add_argument("--all", action="store_true", help="Update everything")
    parser.add_argument("--set-url", metavar="HF_REPO", help="Set Hugging Face model repo")
    parser.add_argument("--status", action="store_true", help="Show update status")
    args = parser.parse_args()

    if args.set_url:
        set_repo_url(args.set_url)
        return

    if args.status:
        show_status()
        return

    if args.all:
        update_all()
        return

    if args.models:
        update_models_from_hf()
        return

    if args.code:
        update_code()
        return

    if args.deps:
        update_deps()
        return

    # Interactive mode
    from rich.console import Console
    import questionary

    console = Console()
    console.print("[bold cyan]MedicalAI - Update Manager[/bold cyan]")
    console.print()

    choice = questionary.select(
        "What would you like to update?",
        choices=[
            "Everything (code + deps + models)",
            "Models only (download latest ONNX)",
            "Code only (git pull)",
            "Dependencies only (pip upgrade)",
            "Show update status",
            "Set Hugging Face model repo",
            "Cancel",
        ],
    ).ask()

    if choice == "Everything (code + deps + models)":
        update_all()
    elif choice == "Models only (download latest ONNX)":
        update_models_from_hf()
    elif choice == "Code only (git pull)":
        update_code()
    elif choice == "Dependencies only (pip upgrade)":
        update_deps()
    elif choice == "Show update status":
        show_status()
    elif "Set Hugging Face" in choice:
        repo = questionary.text("Enter Hugging Face repo (user/repo):").ask()
        if repo:
            set_repo_url(repo)
    else:
        console.print("[yellow]Cancelled.[/yellow]")


if __name__ == "__main__":
    main()
