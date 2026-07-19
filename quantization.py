import argparse
import os

def quantize_fusion():
    from optimize import quantize_fusion as _do
    _do()

def optimize_all():
    from rich.console import Console
    console = Console()
    console.print("[bold cyan]Full Optimization Pipeline[/bold cyan]")
    console.print()
    console.print("[cyan]Quantizing fusion model...[/cyan]")
    from optimize import quantize_fusion as qf
    qf()
    console.print()
    console.print("[green]Optimization complete![/green]")
    console.print("  Fusion classifier: ./checkpoints/onnx/")
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
    print(f"Fusion ONNX:   {'yes' if fusion_onnx else 'no'}")
    print(f"Fusion .pth:   {'yes' if fusion_pt else 'no'}")
    print(f"BLIP model:    {'fine-tuned' if os.path.exists('./blip-xray-finetuned') else 'stock (no fine-tune)'}")

def explain():
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()

    console.print(Panel.fit("[bold cyan]Model Optimization - How It Works[/bold cyan]"))

    console.print()
    console.print("[bold]Two models, two optimization strategies:[/bold]")
    console.print()

    t = Table(title="BLIP (Vision) - Caption Generation Model")
    t.add_column("Setting", style="cyan", width=16)
    t.add_column("What happens", style="white")
    t.add_column("Result", style="green")
    t.add_row(
        "PyTorch (default)",
        "Baseline. Each weight is a 32-bit float (4 bytes). Full math precision.",
        "No setup needed — works immediately"
    )
    t.add_row(
        "FP16 (GPU only)",
        "On GPU the model automatically uses 16-bit floats. Half the memory, faster math.",
        "2x less VRAM, 1.5x speed"
    )
    t.add_row(
        "ONNX export",
        "Requires optimum upgrade: pip install --upgrade optimum. Converts graph to ONNX.",
        "20-30% faster on CPU"
    )
    console.print(t)
    console.print()

    t2 = Table(title="Fusion Model (Symptom Check) - Classifier")
    t2.add_column("Part", style="cyan", width=18)
    t2.add_column("Role", style="white")
    t2.add_column("Optimized how?", style="green")
    t2.add_row(
        "CLIP encoder",
        "Image -> 512 features. Frozen, not trained.",
        "Cached in RAM after first use"
    )
    t2.add_row(
        "Bio_ClinicalBERT",
        "Symptoms -> 768 features. Frozen, not trained.",
        "Cached in RAM after first use"
    )
    t2.add_row(
        "Classifier head",
        "The only trainable part. Tiny MLP: 1280 -> 256 -> N classes. ~0.5 MB.",
        "ONNX export (small gain)"
    )
    console.print(t2)
    console.print()

    console.print("[bold yellow]Recommendations:[/bold yellow]")
    console.print("  1. Run [bold]python quantization.py --mode set-threads[/bold] to keep the system responsive")
    console.print("  2. Run [bold]python quantization.py --mode quantize-fusion[/bold] to optimize Symptom Check")
    console.print("  3. The system uses PyTorch for BLIP (Vision) by default -- no extra steps needed")
    console.print("  4. If you have a GPU, FP16 is automatic -- no extra steps needed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize models for speed and lower memory")
    parser.add_argument("--mode",
                        choices=["quantize-fusion", "optimize-all", "set-threads", "status", "explain"],
                        default="status")
    args = parser.parse_args()

    if args.mode == "quantize-fusion":
        quantize_fusion()
    elif args.mode == "optimize-all":
        optimize_all()
    elif args.mode == "set-threads":
        set_threads()
    elif args.mode == "explain":
        explain()
    else:
        show_status()
