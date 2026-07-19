import importlib
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "images"


def _ensure_dep(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        from rich.console import Console
        console = Console()
        console.print(f"[yellow]'{package_name}' is required for this feature.[/yellow]")
        import questionary
        install = questionary.confirm(f"Install {package_name} now?", default=True).ask()
        if not install:
            return None
        console.print(f"[cyan]Installing {package_name}...[/cyan]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return importlib.import_module(import_name)

def _ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _save_image(pil_image, prefix="capture"):
    _ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    path = str(IMAGES_DIR / filename)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(path, quality=95)
    return path

# ── Camera ──────────────────────────────────────────────────────

def capture_camera():
    cv2 = _ensure_dep("opencv-python", "cv2")
    if cv2 is None:
        return None, "Camera capture requires opencv-python"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "No camera detected (could not open index 0)"

    from rich.console import Console
    console = Console()
    console.print("[cyan]Camera opened. Press SPACE to capture, ESC to cancel.[/cyan]")

    import questionary
    input("Press Enter when ready for camera preview...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "Failed to capture frame from camera"

    preview_path = tempfile.mktemp(suffix="_preview.jpg")
    cv2.imwrite(preview_path, frame)
    preview = Image.open(preview_path)
    os.unlink(preview_path)

    console.print("[cyan]Image captured from camera.[/cyan]")
    path = _save_image(preview, "camera")
    return path, f"Captured from camera -> {path}"

# ── File browser ───────────────────────────────────────────────

def capture_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select an X-ray image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.dcm"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception as e:
        return None, f"File dialog failed: {e}"

    if not path:
        return None, "No file selected"

    return _open_and_save(path)

# ── Manual path entry ──────────────────────────────────────────

def capture_path():
    from rich.console import Console
    console = Console()
    console.print("[cyan]Enter the path to an X-ray image file.[/cyan]")

    import questionary
    path = questionary.path("Image path:").ask()
    if not path:
        return None, "No path entered"
    return _open_and_save(path)

# ── DICOM ──────────────────────────────────────────────────────

def capture_dicom(path=None):
    pydicom = _ensure_dep("pydicom")
    if pydicom is None:
        return None, "DICOM loading requires pydicom"
    np = _ensure_dep("numpy")
    if np is None:
        return None, "DICOM loading requires numpy"

    if not path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Select a DICOM file",
                filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")],
            )
            root.destroy()
        except Exception as e:
            return None, f"File dialog failed: {e}"

    if not path:
        return None, "No DICOM file selected"

    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
        arr = arr - arr.min()
        arr = (arr / arr.max() * 255).astype(np.uint8)
        if len(arr.shape) == 2:
            img = Image.fromarray(arr, mode="L")
            img = ImageOps.equalize(img)
        else:
            img = Image.fromarray(arr)
        result_path = _save_image(img, "dicom")
        return result_path, f"DICOM loaded from {path} -> saved as {result_path}"
    except Exception as e:
        return None, f"Failed to read DICOM: {e}"

# ── Generic open + save ────────────────────────────────────────

def _open_and_save(source_path):
    source_path = str(source_path)
    if source_path.lower().endswith(".dcm"):
        return capture_dicom(source_path)
    try:
        img = Image.open(source_path)
        path = _save_image(img, "import")
        return path, f"Imported from {source_path} -> {path}"
    except Exception as e:
        return None, f"Failed to open image: {e}"

# ── Top-level picker ───────────────────────────────────────────

def pick_image():
    from rich.console import Console
    import questionary

    console = Console()
    method = questionary.select(
        "How do you want to provide the X-ray image?",
        choices=[
            "Browse files on computer",
            "Enter file path manually",
            "Capture from camera",
            "Load DICOM file",
        ],
        pointer=">",
    ).ask()

    result = None
    if method == "Browse files on computer":
        result = capture_file()
    elif method == "Enter file path manually":
        result = capture_path()
    elif method == "Capture from camera":
        result = capture_camera()
    elif method == "Load DICOM file":
        result = capture_dicom()

    return result


if __name__ == "__main__":
    path, msg = pick_image()
    print(msg)
