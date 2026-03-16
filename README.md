# 🎬 Vial Delamination Detection — Video Inspection

A Streamlit web application for **automated vial delamination detection in video footage** using a custom-trained YOLOv8 model. The app processes videos frame-by-frame, tracks vials passing through a defined inspection zone, and classifies each as **Delaminated** or **Non-Delaminated** in real time.

---

## Features

- **Video Upload** — supports MP4, AVI, MOV, MKV, WMV, FLV (up to 500 MB)
- **Live Frame Preview** — annotated frames update in real time during processing
- **Inspection Zone** — a fixed ROI on the left side of the frame simulates a conveyor inspection window
- **Bottle Counter** — counts unique vials entering the inspection zone
- **Zone Detection Panel** — shows the class and confidence of whichever vial is currently in the zone
- **Progress Tracking** — real-time progress bar with frame counter
- **Stop Control** — abort processing at any time
- **Download Annotated Video** — export the fully annotated result as an MP4
- **Adjustable Thresholds** — confidence and IoU sliders in the sidebar

---

## How It Works

```
Upload Video
     │
     ▼
Frame-by-Frame YOLO Inference
     │
     ├── Detection inside Inspection Zone?
     │        ├── YES → classify (Delaminated / Non-Delaminated)
     │        │         draw bounding box + label
     │        │         increment bottle counter on zone entry
     │        └── NO  → mark zone as empty
     │
     ├── Draw zone rectangle on frame
     ├── Write annotated frame to output video
     └── Update live UI (preview, stats, zone panel)
     │
     ▼
Processing Complete
     │
     ├── Show final verdict banner
     ├── Show total frames + bottles inspected
     ├── Show per-class confidence summary
     └── Download annotated MP4
```

### Inspection Zone

The inspection zone is defined as a fixed fraction of the frame dimensions:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ROI_X1_F` | 0.02 | Left edge (2% from left) |
| `ROI_X2_F` | 0.27 | Right edge (27% from left) |
| `ROI_Y1_F` | 0.05 | Top edge (5% from top) |
| `ROI_Y2_F` | 0.95 | Bottom edge (95% from top) |

Only detections whose **bounding box center** falls inside this zone are counted and displayed. This mimics a real conveyor belt inspection camera.

---

## Project Structure

```
video-streamlit/
├── streamlit_app.py        # Main Streamlit application
├── best .pt                # YOLOv8 model weights (required)
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level dependencies (apt)
└── .streamlit/
    └── config.toml         # Theme + upload size config
```

---

## Requirements

### Python dependencies (`requirements.txt`)
| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.32.0 | Web UI framework |
| `ultralytics` | 8.4.21 | YOLOv8 inference |
| `torch` | ≥ 2.0.0 (CPU) | Deep learning backend |
| `torchvision` | ≥ 0.15.0 (CPU) | Vision utilities for torch |
| `opencv-python-headless` | ≥ 4.8.0 | Video decoding & frame annotation |
| `Pillow` | ≥ 10.0.0 | Image handling |
| `numpy` | ≥ 1.24.0 | Array operations |

### System dependencies (`packages.txt`)
| Package | Purpose |
|---------|---------|
| `libgl1` | OpenGL library required by OpenCV |
| `libglib2.0-0t64` | GLib threading library required by OpenCV on Debian Trixie |

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your model in the same folder
#    Make sure 'best .pt' exists in video-streamlit/

# 4. Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## Deploying to Streamlit Cloud

1. Push this folder to a **GitHub repository** (use Git LFS if `best .pt` is > 100 MB)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repository and set **Main file path** to `streamlit_app.py`
4. Click **Deploy**

> **Git LFS setup for large model files:**
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> git add "best .pt"
> git commit -m "add model weights"
> git push
> ```

---

## Usage

1. Open the app in your browser
2. Adjust **Confidence** and **IoU** thresholds in the sidebar (defaults: 0.25 / 0.45)
3. Click **Upload Video** and select a video file
4. Click **▶ Run** to start processing
5. Watch live annotated frames and zone detection stats update in real time
6. Click **■ Stop** to abort early if needed
7. When complete, review the final verdict and download the annotated video

---

## Detection Classes

The model classifies each detected vial into one of two categories:

| Class | Indicator | Description |
|-------|-----------|-------------|
| `Non-Delaminated` | ✅ Green box | Vial glass is intact |
| `Delaminated` | ⚠️ Red/Blue box | Vial shows delamination defects |

---

## Tech Stack

- **[Streamlit](https://streamlit.io)** — interactive web UI
- **[YOLOv8 (Ultralytics)](https://docs.ultralytics.com)** — real-time object detection
- **[OpenCV](https://opencv.org)** — video I/O and frame annotation
- **[PyTorch](https://pytorch.org)** (CPU build) — model inference backend
