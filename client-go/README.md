# MedicalAI Go Clients (TUI + GUI)

Two thin Go clients — a terminal UI and a native desktop GUI — for the
MedicalAI - Light Weight REST API. Neither does any ML work itself: all
inference stays in the Python/PyTorch server, which is where the ML
ecosystem (transformers, ONNX Runtime, the device auto-detection in
`optimize.py`) already lives. These just talk HTTP to that server.

Why a separate Go client at all: it compiles to a single static binary
with no Python/torch runtime, starts in milliseconds, and uses a few MB
of RAM sitting idle — a meaningfully lighter footprint than a second
Python process for anyone who just wants a client on a machine that
isn't running the model itself (e.g. a front-desk workstation talking to
a server elsewhere on the network).

## 1. Start the server

From the repository root (needs the Python environment set up per the
main README):

```bash
python quantization.py --mode serve-api
# -> MedicalAI API Server starting on http://127.0.0.1:8000
```

## 2. Run a client

From this directory:

```bash
# Terminal UI
go run ./cmd/tui

# Desktop GUI
go run ./cmd/gui
```

Both default to `http://127.0.0.1:8000` and accept `--server` to point
elsewhere:

```bash
go run ./cmd/tui --server http://192.168.1.50:8000
```

## Building standalone binaries

```bash
go build -o medicalai-tui ./cmd/tui
go build -o medicalai-gui ./cmd/gui
```

Copy the resulting binary anywhere — no Go toolchain or Python needed on
the target machine, just network access to a running server.

### GUI build dependencies (Linux only)

The GUI (built with [Fyne](https://fyne.io)) uses CGO and links against
your system's graphics libraries. On Debian/Ubuntu:

```bash
sudo apt-get install libgl1-mesa-dev libwayland-dev libx11-dev \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev \
    libxxf86vm-dev libxkbcommon-dev pkg-config
```

macOS and Windows builds need Xcode Command Line Tools / a C compiler
(MinGW/TDM-GCC) respectively — see the
[Fyne getting-started guide](https://docs.fyne.io/started/) for details.
The TUI (`cmd/tui`) has no such requirement — it's pure Go and builds
anywhere the Go toolchain runs.

## Layout

```
client-go/
  internal/api/   Shared HTTP client (client.go) + tests (client_test.go)
  cmd/tui/        Terminal UI (Bubble Tea)
  cmd/gui/        Desktop GUI (Fyne)
```

`internal/api` is the only place that knows the server's HTTP shape;
both front ends are thin views over it. Run `go test ./...` to verify
the client against a mocked server without needing Python/torch
installed at all.

## What's covered

- System health / device / model status
- Vision — generate a radiology caption from an X-ray
- Symptom Check — image + symptom text → diagnosis with confidence,
  urgency, and follow-up guidance
- Medication Interaction Check — screen a medication list for known
  interactions

Every result carries the same disclaimer as the Python UIs: this is
general education, not a medical diagnosis.
