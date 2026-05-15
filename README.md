# AnimForge (`breakdown-animator`)

> **Modern, Zero-Binary 3D Breakdown Image Sequence to Animated GIF & WebP Converter**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Clean](https://img.shields.io/badge/code%20style-clean-brightgreen.svg)](https://github.com/astral-sh/ruff)

`AnimForge` (also known as `breakdown-animator`) is a standalone tool designed specifically for 3D artists, technical artists, game developers, and VFX compositors. It converts sequential render passes (Clay, Wireframe, Albedo, Normal, Roughness, Beauty) and multi-camera breakdowns into optimized looping animated GIFs and high-fidelity WebPs.

Operating with **zero external binary dependencies** (no FFmpeg, ImageMagick, or gifsicle required), AnimForge runs everywhere using pure Python 3.10+ and Pillow, featuring both an interactive dark-mode Textual TUI and a scriptable Rich headless CLI.

---

## 🌟 Key Features

1. **Natural Alphanumeric Sequence Sorting**:
   - Correctly orders unpadded and multi-segment frame filenames (`Step_1`, `Step_2`, ..., `Step_9`, `Step_10`, `Step_11`) without alphabetical sort errors.
2. **Multi-Camera & Render Pass Auto-Grouping**:
   - Automatically detects camera tags (`Camera_Front`, `Camera_Persp`, `Cam_1`) and render passes (`albedo`, `wireframe`, `normal`, `specular`, `clay`) in filenames, partitioning them into clean sequence passes.
3. **Smart Output & Junk Filtering**:
   - Filters out existing output `.gif` and `.webp` animations, system junk (`.DS_Store`, `Thumbs.db`, `desktop.ini`), and non-image artifacts.
4. **ArtStation Auto-Tuning Engine**:
   - Predictor-corrector loop dynamically scales resolution and quantizes palette depth to guarantee target file budgets (e.g. `<= 10.0 MB` for ArtStation, `<= 8.0 MB` for Discord, `<= 5.0 MB` for Slack).
5. **Portfolio 4K 24-Bit WebP Animation**:
   - True 24-bit color fidelity with VP8 lossy and VP8L lossless encoding for crystal-clear web presentations.
6. **End-Frame Freeze Hold Delay**:
   - Configurable pause on the final frame (e.g. `2.0s`) before looping back to the start.
7. **Dimension Harmonization**:
   - Auto-letterboxing with matte padding, proportional fitting, center cropping, and direct stretching for mixed-resolution passes.
8. **Low-RAM Streaming Pipeline**:
   - Generator-based frame processing maintains `< 25MB` peak RAM even across hundreds of 4K render frames.
9. **Dual Interfaces**:
   - **Interactive TUI**: Dark-mode terminal UI built with Textual with live multi-phase progress, sequence inspection, and platform pass/fail summary badges.
   - **Headless CLI**: Scriptable Rich terminal progress and tables for pipeline automation and batch conversions.

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/animforge/breakdown-animator.git
cd breakdown-animator

# Install with pip
pip install -e .
```

---

## 🖥️ Usage

### 1. Interactive Terminal Wizard (Default & Recommended)

Launch the streamlined interactive wizard by running `python app.py` with no arguments (or pass a folder path directly):

```bash
# Launch interactive wizard
python app.py

# Or launch wizard pre-loaded with your breakdown renders folder
python app.py "path/to/renders"
```

**Features**:
- 🎯 **Arrow-Key Selection**: Navigate presets with `↑`/`↓` and hit `Enter`.
- 📁 **Smart Drag & Drop**: Paste file paths directly from Explorer/Finder with automated quote/shell cleaning.
- 🚀 **Live Feedback**: Real-time progress bar with phase indicators (Quantizing, Compressing, Saving).
- 📂 **Instant Actions**: Open folder in File Explorer or copy path to clipboard upon completion.

---

### 2. Full-Screen Textual TUI Mode

For users who prefer a multi-column cockpit dashboard:

```bash
# Launch full-screen Textual TUI
python app.py --tui

# Or launch TUI opening a folder
python app.py "path/to/renders" --tui
```

---

### 3. Bulk & Multi-Folder Batch Processing

Process dozens of render passes or multiple shot folders in a single command or drag-and-drop:

```bash
# Automatically discover and batch-process all subfolders containing render sequences
python app.py "path/to/parent_project" --bulk

# Pass multiple render folders directly
python app.py "renders/shot_01" "renders/shot_02" "renders/shot_03" --output-dir "all_exports"

# Batch process with custom 1.5-second step duration and 3.0-second beauty freeze
python app.py "path/to/parent" --bulk -d 1.5 --hold-last 3.0
```

---

### 4. Headless Command Line Interface (CLI)

Run batch conversions headlessly with Rich progress output:

```bash
# Standard ArtStation Breakdown GIF (1.0s per step, 2.0s hold, <=10MB budget)
python app.py "path/to/renders"

# Set direct duration per breakdown step (e.g. 1.2s per stage)
python app.py "path/to/renders" -d 1.2 --hold-last 2.5

# High-quality Portfolio 4K WebP
python app.py "path/to/renders" --preset portfolio-4k

# Detailed study breakdown (1.8s per step, 3.0s hold)
python app.py "path/to/renders" --preset breakdown-slow

# Smooth 3D rotating model turntable (12 FPS continuous)
python app.py "path/to/renders" --preset turntable

# Target specific camera pass
python app.py "path/to/renders" --camera "Camera_Front"

# Convert all cameras with Discord preset (<=8MB, 0.8s per step)
python app.py "path/to/renders" --camera all --preset discord

# Custom encoding flags
python app.py "path/to/renders" \
  --format gif \
  -d 1.5 \
  --hold-last 2.5 \
  --scale 1080p \
  --max-size-mb 10.0 \
  --no-dither \
  --output-dir "renders/output"

# Dry-run sequence inspection (no files written)
python app.py "path/to/renders" --dry-run
```

---

## 📦 Optimization Presets

| Preset | Target Platform | Format | Step Duration / FPS | Hold Last | Max Size | Notes |
|---|---|---|---|---|---|---|
| `artstation` / `breakdown` | ArtStation Portfolio | GIF | **1.0s / step** | 2.0s | 10.0 MB | Auto-tuned palette & scale |
| `breakdown-slow` | Detail Study | GIF | **1.8s / step** | 3.0s | 10.0 MB | Relaxed pacing for wire/shading |
| `breakdown-fast` | Quick Preview | GIF | **0.5s / step** | 1.5s | 10.0 MB | Snappy stage progression |
| `turntable` | 3D Rotation / Dissolves | GIF | **12.0 FPS** | 1.0s | 10.0 MB | Smooth continuous animation |
| `portfolio-4k` | Portfolio Web | WebP | **1.0s / step** | 2.0s | None | 24-bit VP8 / 92% Quality |
| `discord` | Discord Chat | GIF | **0.8s / step** | 1.5s | 8.0 MB | Optimized for chat embeds |
| `slack` | Slack Workspaces | GIF | **0.8s / step** | 1.0s | 5.0 MB | Compact size budget |
| `custom` | User Configured | GIF/WebP | Custom | Custom | Custom | User-defined parameters |

---

## 🧪 Running the Test Suite

```bash
# Run all unit, integration, CLI, TUI, and bulk/timing tests
pytest -v tests/
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
