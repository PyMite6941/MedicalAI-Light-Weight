"""
Build a standalone .exe for Windows using PyInstaller.
No Python installation needed on the target machine.

Usage:
  python build_exe.py                  # build web_ui.exe
  python build_exe.py --cli            # build run.exe (CLI)
  python build_exe.py --all            # build both

Requires: pip install pyinstaller
"""
import argparse
import os
import shutil
import subprocess
import sys

DIST_DIR = "./dist"


def build_exe(script, name=None):
    if name is None:
        name = os.path.splitext(os.path.basename(script))[0]

    print(f"[cyan]Building {name}.exe from {script}...[/cyan]")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--console",
        "--name", name,
        "--distpath", DIST_DIR,
        "--workpath", "./build",
        "--specpath", "./build",
        "--add-data", "config.json;.",
        script,
    ]

    subprocess.check_call(cmd)

    exe_path = os.path.join(DIST_DIR, f"{name}.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / 1024 / 1024
        print(f"[green]  Created: {exe_path} ({size_mb:.0f} MB)[/green]")
    else:
        print(f"[red]  Failed to create {name}.exe[/red]")

    # Clean up build artifacts
    for d in ["./build", "*.spec"]:
        try:
            if os.path.isdir(d):
                shutil.rmtree(d)
        except Exception:
            pass
    for f in os.listdir("."):
        if f.endswith(".spec"):
            os.remove(f)


def main():
    parser = argparse.ArgumentParser(description="Build standalone executable")
    parser.add_argument("--cli", action="store_true", help="Build CLI executable")
    parser.add_argument("--all", action="store_true", help="Build all executables")
    args = parser.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is required. Install with: pip install pyinstaller")
        sys.exit(1)

    os.makedirs(DIST_DIR, exist_ok=True)

    if args.all:
        build_exe("web_ui.py")
        build_exe("run.py")
        build_exe("batch_predict.py")
    elif args.cli:
        build_exe("run.py")
    else:
        build_exe("web_ui.py")

    print(f"\n[green]Done. Executables in ./{DIST_DIR}/[/green]")
    print("[yellow]Note: The .exe still needs model files (checkpoints/).[/yellow]")
    print("[yellow]Copy the entire project folder to the target machine.[/yellow]")


if __name__ == "__main__":
    main()
