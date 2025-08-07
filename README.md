# CV1 – Real-Time Edge Detection with AI Background Replacement

**Windows 11 + VS Code friendly.**  
MVP‑first implementation that hits all six phases in one script.

---

## What it does

- Captures your webcam feed.
- Runs Canny edge detection with live threshold + blur controls.
- Uses MediaPipe Selfie Segmentation to isolate the person.
- Replaces the background with an image from `backgrounds/`.
- **Edges apply only to you**, not the background.
- View modes: composite, split (original vs. edges), original, edges‑only.
- Press **`s`** to save a screenshot to `outputs/`.

---

## Quick Start (Windows 11, VS Code)

### 1) Clone and open

```powershell
git clone https://github.com/&lt;you&gt;/cv1-realtime-edge-ai-background.git
cd cv1-realtime-edge-ai-background
code .
```

### 2) Create & activate a virtual environment

```powershell
py -m venv .venv
\.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

> If you see a `mediapipe` wheel error, ensure you are on a 64‑bit Python and try:
> ```powershell
> pip install --upgrade pip setuptools wheel
> pip install mediapipe
> ```

### 4) Run it

```powershell
python app.py
```

**Common flags:**

```powershell
python app.py --camera 0 --width 1280 --bg backgrounds\sample_gradient.png
```

- `--camera`: webcam index (0, 1, ...).
- `--width`: output display width (height auto‑scales).
- `--bg`: start with a specific background image.
- `--backgrounds`: folder to load backgrounds from.

On first run, if `backgrounds/` is empty, the app auto‑creates:
- `sample_gradient.png`
- `sample_checker.png`

---

## Controls

- **v** — Toggle view mode (`composite` → `split` → `original` → `edges`)
- **b** — Cycle background (from `backgrounds/`)
- **s** — Save a screenshot to `outputs/`
- **q** — Quit

### Trackbars (window: **Controls**)
- **Canny Low / High** — Lower/upper thresholds
- **Blur (0..10)** — Gaussian blur; slider maps to odd kernel size (1..21)
- **Mask Thresh** — Foreground probability threshold (0.00..1.00)

---

## Folder Structure

```
cv1‑realtime‑edge‑ai‑background/
├─ app.py                # single script with full pipeline (Phases 2–5)
├─ README.md             # setup, usage, controls
├─ requirements.txt      # minimal deps: opencv, mediapipe, numpy
├─ .gitignore            # ignore venv, caches, outputs
├─ backgrounds/          # background images (auto‑seeded on first run)
└─ outputs/              # screenshots saved here by pressing 's'
```

---

## Implementation Notes by Phase

**Phase 1 – Environment & Repo Setup**  
- Uses Python venv, `requirements.txt`, `.gitignore`, `README.md`.

**Phase 2 – Basic Webcam Feed with Canny**  
- `cv2.VideoCapture()` for frames.  
- Grayscale → optional Gaussian blur → Canny.  
- Split‑screen original vs edges (`view mode: split`).

**Phase 3 – AI Segmentation Integration**  
- MediaPipe Selfie Segmentation (`model_selection=1`).  
- Binary mask of person via threshold trackbar.  
- Background replacement from `backgrounds/` folder.

**Phase 4 – Combined Pipeline**  
- Canny applied only to person region.  
- Edges painted onto the person; background stays clean.

**Phase 5 – Parameter Controls**  
- Trackbars: Canny low/high, blur, mask threshold.  
- Background toggle (`b`) and view toggle (`v`).

**Phase 6 – Documentation & Demo Prep**  
- This `README.md` with setup/run instructions.  
- Inline code comments in `app.py`.  
- Press `s` to save screenshots for your portfolio in `outputs/`.

---

## Troubleshooting

- **Black/blank camera window**: Try a different `--camera` index (0 or 1). Close other apps using the webcam.
- **MediaPipe not installing**: Upgrade pip & wheel, ensure 64‑bit Python.
- **Laggy performance**: Reduce `--width` to 960 or 720. Close background apps.
- **Background looks stretched**: The app letterboxes backgrounds to preserve aspect.

---

## License

MIT © You