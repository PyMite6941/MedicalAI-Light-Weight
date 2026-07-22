import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from optimize import (
    clear_memory,
    get_available_models,
    get_device,
    get_memory_usage,
    infer_blip,
    infer_fusion,
    infer_fusion_onnx,
    set_cpu_threads,
)

DEFAULT_MODEL_DIR = "./models/default"
CHECKPOINT_PATH = "./checkpoints/fusion_model.pth"
ONNX_FULL_DIR = "./checkpoints/onnx_full"

def _ensure_default_models():
    if os.path.exists(os.path.join(DEFAULT_MODEL_DIR, "fusion_classifier.onnx")):
        return
    if os.path.exists(CHECKPOINT_PATH):
        return
    print("No models found. Generating default models...")
    subprocess.check_call(
        [sys.executable, "setup_default.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

def _get_model_source():
    models = get_available_models()
    if models["onnx_full_pipeline"]:
        return "ONNX (full pipeline)"
    if models["trained_pytorch"]:
        return "Trained PyTorch"
    if models["default_classifier"]:
        return "Default (random weights)"
    return "NOT AVAILABLE"

# ── FastAPI App (lazy-loaded) ────────────────────────────────

app = None
_executor = None

def _get_app():
    global app, _executor
    if app is not None:
        return app

    import asyncio
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import asynccontextmanager
    from typing import Optional

    from fastapi import FastAPI, File, Form, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    _executor = ThreadPoolExecutor(max_workers=2)

    @asynccontextmanager
    async def _app_lifespan(fapp: FastAPI):
        set_cpu_threads()
        _ensure_default_models()
        yield
        clear_memory()

    a = FastAPI(
        title="MedicalAI - Light Weight API",
        description="REST API for chest X-ray analysis. Upload images for radiology captions or symptom-based diagnosis.",
        version="1.0.0",
        lifespan=_app_lifespan,
    )

    a.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class HealthResponse(BaseModel):
        model_config = {"protected_namespaces": ()}
        status: str
        device: str
        model_source: str
        memory: dict
        models: dict

    class VisionResponse(BaseModel):
        caption: str
        inference_time_ms: float

    class SymptomResponse(BaseModel):
        diagnosis: str
        confidence: float
        inference_time_ms: float

    @a.get("/health", response_model=HealthResponse)
    @a.get("/api/health", response_model=HealthResponse)
    async def health():
        models = get_available_models()
        return HealthResponse(
            status="ok",
            device=get_device().upper(),
            model_source=_get_model_source(),
            memory=get_memory_usage(),
            models=models,
        )

    @a.post("/api/vision", response_model=VisionResponse)
    async def analyze_vision(file: UploadFile = File(...)):
        ext = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            content = await file.read()
            f.write(content)
            path = f.name
        loop = asyncio.get_event_loop()
        try:
            t0 = time.perf_counter()
            caption = await loop.run_in_executor(_executor, infer_blip, path, False)
            elapsed = (time.perf_counter() - t0) * 1000
            return VisionResponse(caption=caption, inference_time_ms=round(elapsed, 1))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            os.unlink(path)
            await loop.run_in_executor(None, clear_memory)

    @a.post("/api/symptom-check", response_model=SymptomResponse)
    async def analyze_symptom(
        file: UploadFile = File(...),
        symptoms: str = Form("No symptoms provided"),
        use_onnx: Optional[bool] = None,
    ):
        ext = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            content = await file.read()
            f.write(content)
            path = f.name
        loop = asyncio.get_event_loop()
        try:
            if use_onnx is None:
                models = get_available_models()
                use_onnx = models.get("onnx_full_pipeline", False)

            t0 = time.perf_counter()
            if use_onnx:
                diagnosis, confidence = await loop.run_in_executor(
                    _executor, infer_fusion_onnx, path, symptoms
                )
            else:
                diagnosis, confidence = await loop.run_in_executor(
                    _executor, infer_fusion, path, symptoms
                )
            elapsed = (time.perf_counter() - t0) * 1000

            if diagnosis is None:
                raise HTTPException(status_code=500, detail=confidence)

            return SymptomResponse(
                diagnosis=diagnosis,
                confidence=round(confidence, 4),
                inference_time_ms=round(elapsed, 1),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            os.unlink(path)
            await loop.run_in_executor(None, clear_memory)

    @a.get("/api/models")
    async def list_models():
        return get_available_models()

    app = a
    return app

# ── Original CLI Modes ──────────────────────────────────────

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

def set_threads():
    from optimize import set_cpu_threads as sct, get_memory_usage as gmu
    n = sct()
    mem = gmu()
    print(f"CPU threads set to {n}")
    print(f"Current RAM: {mem['rss_mb']:.0f} MB")

def show_status():
    import torch
    mem = get_memory_usage()
    print(f"Device:        {get_device().upper()}")
    print(f"FP16 mode:     {'ON' if get_device() in ('cuda', 'mps') else 'OFF'}")
    print(f"Process RAM:   {mem['rss_mb']:.0f} MB")
    print(f"Torch threads: {torch.get_num_threads()}")
    fusion_onnx = os.path.exists("./checkpoints/onnx/fusion_classifier.onnx")
    fusion_pt = os.path.exists(CHECKPOINT_PATH)
    fusion_full = os.path.exists(ONNX_FULL_DIR + "/fusion_full.onnx")
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
    console.print("[bold]Deployment options:[/bold]")
    console.print()
    t = Table(title="Deployment Options")
    t.add_column("Option", style="cyan", width=22)
    t.add_column("What it is", style="white")
    t.add_column("Best for", style="green")
    t.add_row("PyTorch (default)", "Full model in PyTorch.", "Development, GPU users")
    t.add_row("ONNX Classifier", "Exports classifier head only.", "Minor CPU speedup")
    t.add_row("ONNX Full Pipeline", "Exports entire pipeline to ONNX.", "Production, no-PyTorch")
    console.print(t)
    console.print()
    t2 = Table(title="Fusion Model Components")
    t2.add_column("Part", style="cyan", width=18)
    t2.add_column("Role", style="white")
    t2.add_column("Size", style="green")
    t2.add_row("CLIP encoder", "Image -> 512 features", "~600 MB")
    t2.add_row("Bio_ClinicalBERT", "Symptoms -> 768 features", "~400 MB")
    t2.add_row("Classifier head", "1280 -> 256 -> N classes", "~0.5 MB")
    console.print(t2)

# ── API Server Mode ─────────────────────────────────────────

def serve_api(host="127.0.0.1", port=8000, reload=False):
    global app
    _get_app()
    import uvicorn
    print(f"MedicalAI API Server starting on http://{host}:{port}")
    print(f"Device: {get_device().upper()}")
    print(f"Docs:   http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/health")
    print()
    print("Your website can connect to this API using the endpoints above.")
    uvicorn.run("quantization:app", host=host, port=port, reload=reload)

# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedicalAI - Export & Web API server")
    parser.add_argument("--mode",
                        choices=[
                            "quantize-fusion", "export-full", "optimize-all",
                            "set-threads", "status", "explain",
                            "serve-api",
                        ],
                        default="status")
    parser.add_argument("--host", default="127.0.0.1", help="Host for serve-api (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for serve-api (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload for development")
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
    elif args.mode == "serve-api":
        serve_api(host=args.host, port=args.port, reload=args.reload)
    else:
        show_status()
