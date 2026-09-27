# How the Fetal Anomaly Detection Website Was Built: Engineering & Implementation Breakdown

This document provides an in-depth technical walkthrough of how the **Fetal Anomaly Detection System** web application was engineered, designed, structured, and verified from the ground up.

---

## 1. Architectural Philosophy & Technology Stack

The project was designed with four primary engineering requirements:
1. **Zero Simulated/Mock Data:** The website must connect directly to the real trained PyTorch ResNet50 model (`best_resnet50.pth`) and dynamic class mapping (`class_mapping.json`).
2. **Decoupled Client-Server Architecture:** A high-performance Python FastAPI backend provides REST endpoints for heavy GPU/CPU tensor computations and Grad-CAM generation, while a modern Next.js 15 / React 19 / TypeScript frontend handles state management, accessibility, and UI rendering.
3. **Medical & Research-Grade UI/UX:** A clean, professional aesthetic (slate/teal/cyan palette, soft borders, zero gaming-style gradients) adhering to academic and clinical standards.
4. **Data Privacy & Local Execution:** Ultrasound scans are processed locally in-memory without sending patient images to external cloud APIs or third-party services.

```text
┌────────────────────────────────────────────────────────┐
│                   Next.js 15 Frontend                  │
│  - App Router (React 19 Server & Client Components)    │
│  - TypeScript (Strict Type Contracts)                  │
│  - Tailwind CSS (Curated Medical Theme)                │
│  - Lucide React (Clinical & System Icons)              │
└───────────────────────────┬────────────────────────────┘
                            │
               JSON / HTTP (CORS Enabled)
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                    FastAPI Backend                     │
│  - Lifespan Singleton Model Management                │
│  - PyTorch ResNet50 Inference Engine                   │
│  - Dynamic Class Mapper (16 Categories)                │
│  - Grad-CAM Hook Engine (layer4[-1].conv3)             │
│  - OpenCV & Pillow Image Preprocessing                 │
└────────────────────────────────────────────────────────┘
```

---

## 2. Backend Engineering (FastAPI + PyTorch)

The backend is structured into specialized, decoupled modules within `backend/`:

### 2.1 Configuration & Path Safety (`backend/config.py`)
- Utilizes Python's `pathlib.Path` to guarantee cross-platform compatibility across Windows, Linux, and macOS without hardcoded backslashes (`\`).
- Exposes environment variables for:
  - `MODEL_PATH`: Location of the trained weights file (`models/best_resnet50.pth`).
  - `CLASS_MAPPING_PATH`: Location of `config/class_mapping.json`.
  - `CONFIDENCE_THRESHOLD`: Configurable threshold (default `0.60`) for flagging low-confidence predictions.
  - `CORS_ORIGINS`: Cross-Origin Resource Sharing whitelist (default `http://localhost:3000`).

### 2.2 Singleton Model Loading & Lifecycle (`backend/model.py` & `backend/main.py`)
- **Problem:** Loading a 90MB deep neural network on every HTTP request causes severe latency (1.5–3 seconds per request) and memory leaks.
- **Solution:** A singleton class `FetalAnomalyModel` is instantiated once and managed by FastAPI's `lifespan` async context manager.
- **Dynamic Reconstruction:**
  1. Parses `class_mapping.json` at startup to determine the exact number of classes ($N = 16$).
  2. Constructs `resnet50(weights=None)`.
  3. Replaces the linear classification head: `model.fc = nn.Linear(model.fc.in_features, 16)`.
  4. Loads state dictionary via `torch.load(weights_path, map_location=device)`.
  5. Sets the network to evaluation mode (`model.eval()`).
  6. Automatically selects CUDA if an NVIDIA GPU is present, otherwise falls back gracefully to CPU.

### 2.3 Deterministic Inference Preprocessing (`backend/preprocessing.py`)
To prevent domain shift between evaluation and production, preprocessing strictly replicates the training pipeline:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```
Uploaded images of any format (grayscale, RGBA, CMYK) are converted to 3-channel RGB before applying the PyTorch transform.

### 2.4 Grad-CAM Visual Explainability Engine (`backend/gradcam.py`)
- Implements **Gradient-weighted Class Activation Mapping** targeting the final convolutional layer: `model.layer4[-1].conv3`.
- **PyTorch Hook Architecture:**
  - Forward hook captures activations $A \in \mathbb{R}^{1 \times 2048 \times 7 \times 7}$.
  - Full backward hook captures gradients $\frac{\partial Y^c}{\partial A}$ during backpropagation of the predicted class score $Y^c$.
- **Mathematical Computation:**
  1. Computes global average pooling on gradients to calculate channel importance weights:
     $$\alpha_k = \frac{1}{H \times W} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
  2. Multiplies weights with forward activations and applies ReLU:
     $$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
  3. Normalizes heatmap values to $[0, 1]$ and resizes to $224 \times 224$ via OpenCV.
  4. Maps values to `cv2.COLORMAP_JET` and creates an alpha-blended overlay ($60\%$ ultrasound image $+ 40\%$ colorized heatmap).
  5. Saves unique images to `backend/outputs/gradcam/` and serves them statically via FastAPI.

### 2.5 Security, Sanitization & Integrity Validation (`backend/utils.py`)
- **File Type & MIME Validation:** Enforces whitelist `.jpg`, `.jpeg`, `.png` and MIME types `image/jpeg`, `image/png`.
- **Payload Size Capping:** Limits uploaded files to 10 MB to prevent denial-of-service (DoS) memory exhaustion.
- **Corrupted File Detection:** Opens files in Pillow and executes `img.verify()` to catch corrupted or malformed byte streams before passing them to PyTorch.
- **Path Traversal Protection:** Sanitizes filenames using regular expressions and generates randomized UUID tokens (`gradcam_heatmap_{uuid}.png`) to prevent directory traversal or file overwrites.
- **Automatic Output Garbage Collection:** Automatically prunes old Grad-CAM images when output directory exceeds 200 files.

### 2.6 REST API Endpoints (`backend/main.py` & `backend/schemas.py`)
- `GET /health`: Returns operational status, active device (`cuda` or `cpu`), model loaded status, and class count.
- `GET /model-info`: Returns model name, input resolution ($224 \times 224$), target layer (`layer4[-1].conv3`), and ImageNet normalization stats.
- `POST /predict`: Receives multipart form data (`file`), validates and preprocesses the image, runs ResNet50 inference, computes Grad-CAM, and returns a type-safe JSON payload.

---

## 3. Frontend Engineering (Next.js 15 + TypeScript + Tailwind)

The frontend is built using Next.js 15 with the React App Router, structured into reusable, single-responsibility components:

### 3.1 Type-Safe API Client (`frontend/lib/api.ts`)
- Centralizes all HTTP communication with the FastAPI backend.
- Declares strict TypeScript interfaces matching the backend's Pydantic schemas:
  - `PredictResponse`
  - `ClassProbability`
  - `PredictionResult`
  - `GradCAMResult`
  - `HealthResponse`
  - `ModelInfoResponse`
- Reads `NEXT_PUBLIC_API_URL` from `.env.local` (defaulting to `http://localhost:8000`).

### 3.2 Modular Component Breakdown (`frontend/components/`)
1. **`Navbar.tsx`**: Responsive header featuring project identity, navigation links (`Home`, `Analyze`, `How It Works`, `About`, `Disclaimer`), mobile menu drawer, and blurred glassmorphism backdrop.
2. **`Footer.tsx`**: Academic footer with project title, technology credits, and prominent non-clinical disclaimer badge.
3. **`HeroSection.tsx`**: Interactive landing hero featuring title typography, direct action buttons, and a visual 5-step AI pipeline diagram with gradient step cards.
4. **`UploadCard.tsx`**:
   - Supports drag-and-drop and native file selection.
   - Real-time client-side validation (MIME, file extension, max 10MB).
   - Immediate image preview rendering via JavaScript `FileReader`.
   - File info badge (filename, size in KB) with remove/replace buttons.
5. **`LoadingState.tsx`**:
   - Animated medical spinner.
   - Staggered pulse indicators displaying the active stage (*Preprocessing*, *ResNet50 Inference*, *Grad-CAM Generation*).
6. **`PredictionCard.tsx`**:
   - Large display of the predicted anomaly category.
   - Dynamic confidence progress bar with color-coded severity:
     - $\ge 80\%$: Teal / Cyan gradient
     - $60\% - 79\%$: Blue / Indigo gradient
     - $< 60\%$: Amber / Orange warning gradient
   - Low-confidence alert notice when confidence falls below the configured threshold.
   - Compact metadata grid showing Model (`ResNet50`), Resolution (`224 × 224`), Explainability (`Grad-CAM`), and Class Index.
7. **`ProbabilityList.tsx`**:
   - Displays all 16 classes returned dynamically from the model.
   - Automatically sorted by highest probability first.
   - Highlights the top predicted class with active border and indicator dot.
   - Proportional progress bars for quick visual scanning.
8. **`GradCAMViewer.tsx`**:
   - Visual explainability module for inspecting neural network focus.
   - Interactive tab selector: **Original Ultrasound**, **Grad-CAM Heatmap**, and **Explainability Overlay**.
   - **Interactive Opacity Slider:** Real-time range slider (0% to 100%) dynamically blending the overlay over the raw ultrasound.
   - **Side-by-Side 3-Panel Matrix:** Desktop view showing Original, Heatmap, and Overlay simultaneously.
9. **`ErrorMessage.tsx`**:
   - User-friendly error card that hides raw Python stack traces.
   - Actionable "Try Again" retry trigger.
10. **`DisclaimerBanner.tsx`**:
    - Consistent medical advisory banner warning that the system is an AI research prototype.

### 3.3 Page Route Structure (`frontend/app/`)
- `/`: **Home Landing Page** — Overview of capabilities, pipeline diagram, feature cards, and research overview.
- `/analyze`: **Main Analysis Workspace** — End-to-end interactive workflow for uploading ultrasound images, running inference, and inspecting Grad-CAM overlays.
- `/how-it-works`: **Detailed Pipeline Guide** — 6-step deep dive into ultrasound sonography preprocessing, convolutional neural network inference, and Grad-CAM mathematics.
- `/about`: **Research Project Documentation** — Project background, technical architecture specifications, and real verified evaluation metrics (82.95% accuracy, 0.9732 AUC).
- `/disclaimer`: **Medical Advisory & Regulatory Boundaries** — Outlines clinical limitations, educational scope, and data privacy policies.

---

## 4. UI/UX Design System & Aesthetics

1. **Curated Color Palette:**
   - **Backgrounds:** Slate-50 / Pure White for a modern, clinical look.
   - **Primary Accents:** Deep Teal (`#0d9488`) and Bright Cyan (`#06b6d4`) representing precision and intelligence.
   - **Neutral Structure:** Slate-200 borders and Slate-600/900 typography for high contrast and readability.
   - **Advisories:** Soft Amber (`#f59e0b`) and Rose (`#f43f5e`) for warnings and disclaimers without visual clutter.
2. **Typography:**
   - Modern `Inter` sans-serif loaded from Google Fonts via Next.js Font Optimization (`next/font/google`).
3. **Accessibility (a11y):**
   - High-contrast text meeting WCAG AA standards.
   - Semantic HTML5 landmark tags (`<main>`, `<nav>`, `<section>`, `<footer>`).
   - Explicit `aria-label` tags on buttons, sliders, and interactive file pickers.
   - Custom `focus-visible` styling for full keyboard navigation.
4. **Responsive Layouts:**
   - Full mobile-first responsiveness using Tailwind CSS grid and flex utilities.
   - Desktop side-by-side diagnostic cards gracefully stack into single-column layouts on mobile viewports.

---

## 5. Testing, Verification & Quality Assurance

### 5.1 Automated Backend Suite (`test_backend.py`)
An automated test suite was constructed and run against the real weights:
- Verified that `best_resnet50.pth` (90.10 MB) exists and parses without errors.
- Verified that `class_mapping.json` loads all 16 classes dynamically.
- Tested tensor transformations (`[1, 3, 224, 224]`).
- Executed real forward inference on test ultrasound images from the dataset.
- Verified forward and backward hook execution on `layer4[-1].conv3`.
- Tested FastAPI endpoints (`/health`, `/model-info`, `/predict`) using `fastapi.testclient.TestClient`.

### 5.2 Next.js Production Build Verification
The frontend was compiled and built with Next.js Turbopack:
```powershell
npm run build
```
- **Result:** `✓ Compiled successfully in 31.5s` with `0 TypeScript errors` and `0 ESLint warnings`.
- All static routes (`/`, `/about`, `/analyze`, `/disclaimer`, `/how-it-works`) were prerendered and optimized.

---

## 6. How to Maintain & Extend the System

1. **Updating or Retraining the Model:**
   - If you train a new model or adjust hyperparameters, simply replace `fetal-anomaly-ai/backend/models/best_resnet50.pth`.
   - If the class list changes, update `fetal-anomaly-ai/backend/config/class_mapping.json`.
   - The backend dynamically adjusts `model.fc` and class count automatically upon restart.
2. **Adjusting the Confidence Threshold:**
   - Change `CONFIDENCE_THRESHOLD=0.60` in `backend/config.py` or provide it as an environment variable in production.
3. **Production Deployment:**
   - **Backend:** Can be containerized with Docker or deployed on AWS EC2 / Azure VM / GCP Compute Engine using `uvicorn main:app --workers 4`.
   - **Frontend:** Can be deployed to Vercel, Netlify, or Node.js Docker containers by pointing `NEXT_PUBLIC_API_URL` to your production backend domain.
