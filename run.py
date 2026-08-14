import os

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import capture
from bone_fracture import infer_bone_fracture, model_available as bone_model_available
from drug_interactions import LEVEL_FROM_STR, check_medication_list, db_available, format_interaction
from drug_interactions import DISCLAIMER as DRUG_DISCLAIMER
from knowledge import get_condition_info, DISCLAIMER
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

    info = get_condition_info(diagnosis)
    console.print()
    console.print(f"[bold]About this finding:[/bold] {info['description']}")
    if info["common_symptoms"]:
        console.print(f"[bold]Commonly associated with:[/bold] {', '.join(info['common_symptoms'])}")
    console.print(f"[bold]Urgency:[/bold] {info['urgency']}  [bold]Suggested next step:[/bold] {info['follow_up']}")
    console.print(f"[dim]{DISCLAIMER}[/dim]")

    return diagnosis, confidence

def run_bone_check(image_path):
    console.print("[cyan]Running bone X-ray fracture screen (experimental)...[/cyan]")
    findings, confidences = infer_bone_fracture(image_path)

    if findings is None:
        console.print(f"[yellow]{confidences}[/yellow]")
        return

    if findings:
        table = Table(title="Bone X-ray Screening Result")
        table.add_column("Finding", style="cyan")
        table.add_column("Confidence", style="green")
        for label in findings:
            table.add_row(label, f"{confidences[label]:.1%}")
        console.print(table)
        for label in findings:
            info = get_condition_info(label)
            console.print(f"[bold]{label}:[/bold] {info['description']} [dim](urgency: {info['urgency']})[/dim]")
    else:
        console.print("[green]No fracture or hardware detected above threshold.[/green]")

    console.print(f"[dim]All confidences: {', '.join(f'{k}={v:.1%}' for k, v in confidences.items())}[/dim]")
    console.print(f"[bold yellow]{DISCLAIMER}[/bold yellow]")
    return findings, confidences

def run_drug_check():
    console.print("[cyan]Enter the patient's current medications, one per line. Blank line to finish.[/cyan]")
    meds = []
    while True:
        line = questionary.text(f"Medication {len(meds) + 1} (blank to finish):").ask()
        if not line:
            break
        meds.append(line)

    if len(meds) < 2:
        console.print("[yellow]Need at least 2 medications to check for interactions.[/yellow]")
        return

    interactions, unrecognized = check_medication_list(meds)

    if interactions:
        table = Table(title="Potential Interactions")
        table.add_column("Drug A", style="cyan")
        table.add_column("Drug B", style="cyan")
        table.add_column("Severity", style="yellow")
        table.add_column("Source", style="dim")
        for r in sorted(interactions, key=lambda x: LEVEL_FROM_STR.get(x["level"], 0), reverse=True):
            table.add_row(r["drug_a"], r["drug_b"], r["level"], r["source"])
        console.print(table)
        for r in interactions:
            if r.get("note"):
                console.print(f"  [dim]{r['drug_a']} + {r['drug_b']}: {r['note']}[/dim]")
    else:
        console.print("[green]No known interactions found among these medications.[/green]")

    if unrecognized:
        console.print(f"[dim]Not found in the interaction database: {', '.join(unrecognized)}[/dim]")
    if not db_available():
        console.print("[dim]Using the small built-in reference set only. Run 'python drug_interactions.py --download' for much broader coverage (needs network, one-time).[/dim]")
    console.print(f"[bold yellow]{DRUG_DISCLAIMER}[/bold yellow]")

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
                "Bone X-ray — Check for fracture (experimental)" + ("" if bone_model_available() else " [not trained yet]"),
                "Medication Interaction Check",
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

        if choice == "Medication Interaction Check":
            run_drug_check()
            console.print()
            again = questionary.confirm("Do another?").ask()
            if not again:
                break
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
        elif choice.startswith("Bone X-ray"):
            run_bone_check(image_path)

        clear_memory()
        console.print()
        again = questionary.confirm("Do another?").ask()
        if not again:
            break

    clear_memory()

if __name__ == "__main__":
    main()
