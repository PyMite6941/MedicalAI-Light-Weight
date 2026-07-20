import argparse
import os

def quantize_fusion():
    from optimize import quantize_fusion as _do
    _do()

def export_full():
    from optimize import export_full_fusion_onnx
    export_full_fusion_onnx()

def optimize_all():
    from rich.console import Console
    console = Console()
    console.print("[bold cyan]Full Optimization Pipeline[/bold cyan]")
    console.print()
    console.print("[cyan]Quantizing fusion model...[/cyan]")
    from optimize import quantize_fusion as qf
    qf()
    console.print()
    console.print("[cyan]Exporting full ONNX pipeline...[/cyan]")
    from optimize import export_full_fusion_onnx as ef
    ef()
    console.print()
    console.print("[green]Optimization complete![/green]")
    console.print("  Fusion classifier: ./checkpoints/onnx/fusion_classifier.onnx")
    console.print("  Full pipeline:     ./checkpoints/onnx_full/fusion_full.onnx")
    console.print("[yellow]  BLIP ONNX skipped — upgrade optimum to enable: pip install --upgrade optimum[/yellow]")

def set_threads():
    from optimize import set_cpu_threads, get_memory_usage
    n = set_cpu_threads()
    mem = get_memory_usage()
    print(f"CPU threads set to {n}")
    print(f"Current RAM: {mem['rss_mb']:.0f} MB")

def show_status():
    from optimize import get_memory_usage, get_device, use_fp16
    import torch

    mem = get_memory_usage()
    print(f"Device:        {get_device().upper()}")
    print(f"FP16 mode:     {'ON' if use_fp16() else 'OFF'}")
    print(f"Process RAM:   {mem['rss_mb']:.0f} MB")
    print(f"Torch threads: {torch.get_num_threads()}")

    fusion_onnx = os.path.exists("./checkpoints/onnx/fusion_classifier.onnx")
    fusion_pt = os.path.exists("./checkpoints/fusion_model.pth")
    fusion_full = os.path.exists("./checkpoints/onnx_full/fusion_full.onnx")
    print(f"Fusion ONNX (classifier): {'yes' if fusion_onnx else 'no'}")
    print(f"Fusion ONNX (full):       {'yes' if fusion_full else 'no'}")
    print(f"Fusion .pth:              {'yes' if fusion_pt else 'no'}")
    print(f"BLIP model:    {'fine-tuned' if os.path.exists('./blip-xray-finetuned') else 'stock (no fine-tune)'}")

def explain():
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()

    console.print(Panel.fit("[bold cyan]Model Optimization - How It Works[/bold cyan]"))

    console.print()
    console.print("[bold]Three deployment options:[/bold]")
    console.print()

    t = Table(title="Deployment Options Comparison")
    t.add_column("Option", style="cyan", width=22)
    t.add_column("What it is", style="white")
    t.add_column("Best for", style="green")
    t.add_row(
        "PyTorch (default)",
        "Full model in PyTorch. Everything works immediately.",
        "Development, testing, GPU users"
    )
    t.add_row(
        "ONNX Classifier",
        "Exports just the tiny classifier head to ONNX. Encoders stay in PyTorch.",
        "Minor CPU speedup, backward compat"
    )
    t.add_row(
        "ONNX Full Pipeline",
        "Exports entire image+text pipeline to a single ONNX graph.",
        "Production deployment, no-PyTorch, CI/CD pipelines"
    )
    console.print(t)
    console.print()

    t2 = Table(title="Fusion Model (Symptom Check) - Components")
    t2.add_column("Part", style="cyan", width=18)
    t2.add_column("Role", style="white")
    t2.add_column("Size", style="green")
    t2.add_row("CLIP encoder", "Image -> 512 features", "~600 MB")
    t2.add_row("Bio_ClinicalBERT", "Symptoms -> 768 features", "~400 MB")
    t2.add_row("Classifier head", "1280 -> 256 -> N classes", "~0.5 MB")
    console.print(t2)
    console.print()

    console.print("[bold yellow]Recommendations for lower-tech setups:[/bold yellow]")
    console.print("  1. Run [bold]python quantization.py --mode export-full[/bold] (standalone ONNX)")
    console.print("  2. Run [bold]python web_ui.py[/bold] (browser-based GUI, no CLI needed)")
    console.print("  3. Or run [bold]launch.ps1[/bold] (one-click setup + launch)")
    console.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize models for speed and lower memory")
    parser.add_argument("--mode",
                        choices=["quantize-fusion", "export-full", "optimize-all", "set-threads", "status", "explain"],
                        default="status")
    args = parser.parse_args()

    if args.mode == "quantize-fusion":
        quantize_fusion()
    elif args.mode == "export-full":
        export_full()
    elif args.mode == "optimize-all":
        optimize_all()
    elif args.mode == "set-threads":
        set_threads()
    elif args.mode == "explain":
        explain()
    else:
        show_status()
