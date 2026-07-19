import os

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import capture
from optimize import (
    clear_memory,
    get_device,
    get_memory_usage,
    infer_blip,
    infer_fusion,
    set_cpu_threads,
)

console = Console()

def run_vision(image_path):
    console.print("[cyan]Running Vision analysis...[/cyan]")
    try:
        caption = infer_blip(image_path)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    console.print()
    console.print(Panel(f"[bold green]{caption}[/bold green]", title="Generated Radiology Caption"))
    return caption

def run_symptom_check(image_path):
    symptoms = questionary.text("Enter patient symptoms / clinical indication:").ask()
    if not symptoms:
        symptoms = "No symptoms provided"

    try:
        diagnosis, confidence = infer_fusion(image_path, symptoms)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    if diagnosis is None:
        console.print(f"[red]{confidence}[/red]")
        return

    table = Table(title="Diagnosis Result")
    table.add_column("Prediction", style="cyan")
    table.add_column("Confidence", style="green")
    table.add_row(diagnosis, f"{confidence:.1%}")
    console.print(table)

    if confidence < 0.75:
        console.print("[yellow]Low confidence — consider follow-up.[/yellow]")

    return diagnosis, confidence

def show_status():
    from training import info as training_info

    console.print("[bold cyan]--- System Status ---[/bold cyan]")
    mem = get_memory_usage()
    console.print(f"Device:        [green]{get_device().upper()}[/green]")
    console.print(f"Process RAM:   {mem['rss_mb']:.0f} MB")
    console.print()

    console.print("[bold cyan]--- Fusion Model Data ---[/bold cyan]")
    training_info()
    console.print()

    cap_dir = capture.IMAGES_DIR
    count = len(list(cap_dir.glob("*"))) if cap_dir.exists() else 0
    console.print(f"Captured images: {count} in {cap_dir}")

def install_deps():
    deps = []
    try:
        import cv2
    except ImportError:
        deps.append("opencv-python")
    try:
        import pydicom
    except ImportError:
        deps.append("pydicom")
    try:
        import nltk
    except ImportError:
        deps.append("nltk")

    if not deps:
        console.print("[green]All optional dependencies are already installed.[/green]")
        return

    import subprocess
    import sys
    console.print(f"[yellow]Installing: {' '.join(deps)}[/yellow]")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
    console.print("[green]Done.[/green]")

def main():
    set_cpu_threads()

    console.print(Panel.fit("[bold cyan]MedicalAI - Light Weight[/bold cyan]"))
    console.print()

    while True:
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                "Vision — Generate report from X-ray",
                "Symptom Check — Diagnose from X-ray + symptoms",
                "System Status & Data Info",
                "Install optional deps (camera, DICOM, BLEU)",
                "Exit",
            ],
            pointer=">",
        ).ask()

        if choice == "Exit":
            console.print("[bold red]Exiting...[/bold red]")
            clear_memory()
            break

        if choice == "Install optional deps (camera, DICOM, BLEU)":
            install_deps()
            continue

        if choice == "System Status & Data Info":
            show_status()
            console.print()
            continue

        image_path, msg = capture.pick_image()
        if image_path is None:
            console.print(f"[red]{msg}[/red]")
            continue
        console.print(f"[dim]{msg}[/dim]")

        if choice.startswith("Vision"):
            run_vision(image_path)
        elif choice.startswith("Symptom Check"):
            run_symptom_check(image_path)

        clear_memory()
        console.print()
        again = questionary.confirm("Do another?").ask()
        if not again:
            break

    clear_memory()

if __name__ == "__main__":
    main()
