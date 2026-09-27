# Fetal Anomaly Detection System
### AI-Assisted Analysis of Fetal Ultrasound Images Using Deep Learning

> **Final-Year Research Project:** *Early Detection of Fetal Anomalies During Pregnancy Using Deep Learning*  
> **Architecture:** ResNet50 Convolutional Neural Network + Grad-CAM Visual Explainability  
> **Stack:** PyTorch, Torchvision, FastAPI, Next.js 15, TypeScript, Tailwind CSS, Lucide React

---

## 🔬 System Overview

The **Fetal Anomaly Detection System** is an end-to-end medical deep learning platform engineered to perform automated multi-class classification on prenatal ultrasound sonograms. Utilizing a fine-tuned **ResNet50** convolutional neural network, the platform computes class probability distributions across 16 anomaly categories and generates **Grad-CAM** (Gradient-weighted Class Activation Mapping) explainability heatmaps directly from the network's final convolutional layer (`layer4[-1].conv3`).

---

## 📐 System Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   Next.js 15 Frontend                  │
│       (React 19, TypeScript, Tailwind CSS, Lucide)     │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST (multipart/form-data)
                            ▼
┌────────────────────────────────────────────────────────┐
│                    FastAPI Backend                     │
│               (Lifespan Model Singleton)               │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               Image Validation & Sanitization          │
│       (Size, MIME, Extension, Corrupted File Check)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               Inference Preprocessing Pipeline         │
│  RGB Convert → Resize(224, 224) → ToTensor → Normalize │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                PyTorch ResNet50 Backbone               │
│               Weights: best_resnet50.pth               │
│               Mapping: class_mapping.json              │
└─────────────┬────────────────────────────┬─────────────┘
              │ Logits                     │ Backward Hooks
              ▼                            ▼
┌───────────────────────────┐┌───────────────────────────┐
│   Softmax Probabilities   ││         Grad-CAM          │
│     & Top Prediction      ││  (layer4[-1].conv3 CAM)  │
└─────────────┬─────────────┘└─────────────┬─────────────┘
              │                            │
              └─────────────┬──────────────┘
                            │ Structured JSON + Image URL
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Results & Analysis UI                 │
│  - Top Predicted Class & Confidence Bar                │
│  - Sorted Probability Breakdown for all 16 Classes     │
│  - Interactive Original / Heatmap / Overlay Inspector  │
│  - Opacity Slider & Side-by-Side Visualizer            │
└────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Directory Structure

```text
fetal-anomaly-ai/
├── backend/
│   ├── main.py                # FastAPI endpoints (/health, /model-info, /predict)
│   ├── model.py               # ResNet50 loader singleton & inference pipeline
│   ├── preprocessing.py       # Exact PyTorch preprocessing (Resize, ToTensor, Normalize)
│   ├── gradcam.py             # Grad-CAM forward/backward hooks & overlay generator
│   ├── schemas.py             # Pydantic response & request data contracts
│   ├── config.py              # Environment configuration & pathlib definitions
│   ├── utils.py               # File validation, sanitization, safe naming
│   ├── requirements.txt       # Python dependencies
│   ├── models/
│   │   └── best_resnet50.pth  # Real trained PyTorch weights (16 classes)
│   ├── config/
│   │   └── class_mapping.json # Dynamic class-to-index mapping
│   └── outputs/
│       └── gradcam/           # Generated Grad-CAM heatmaps & overlays
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx           # Landing page with visual AI pipeline
│   │   ├── analyze/page.tsx   # Ultrasound upload, preview & analysis interface
│   │   ├── how-it-works/page.tsx # Detailed 6-step deep learning pipeline explanation
│   │   ├── about/page.tsx     # Research project details & evaluation metrics
│   │   ├── disclaimer/page.tsx# Formal medical advisory & operational scope
│   │   ├── layout.tsx         # Global layout with Inter font & metadata
│   │   └── globals.css        # Tailwind styling & slider controls
│   ├── components/
│   │   ├── Navbar.tsx         # Responsive navigation header
│   │   ├── Footer.tsx         # Footer with research credits & disclaimer
│   │   ├── HeroSection.tsx    # Landing page hero & animated pipeline
│   │   ├── UploadCard.tsx     # Drag & drop image uploader with preview
│   │   ├── LoadingState.tsx   # Multi-stage inference progress indicator
│   │   ├── PredictionCard.tsx # Prediction heading, confidence bar & specs
│   │   ├── ProbabilityList.tsx# Sorted dynamic class probability bars
│   │   ├── GradCAMViewer.tsx  # Interactive tabs, slider & side-by-side view
│   │   ├── ErrorMessage.tsx   # Friendly error display with retry action
│   │   └── DisclaimerBanner.tsx # Prominent research prototype notice
│   ├── lib/
│   │   └── api.ts             # Centralized typed API client
│   ├── .env.local             # Frontend environment variables
│   └── package.json
│
├── .gitignore
└── README.md
```

---

## 🧠 Model Integration & Preprocessing

### Dynamic Model Reconstruction
The network architecture is dynamically constructed at startup based on the number of classes defined in `class_mapping.json`:

```python
from torchvision.models import resnet50
import torch.nn as nn
import torch

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load("models/best_resnet50.pth", map_location=device))
model.to(device)
model.eval()
```

### Deterministic Inference Preprocessing
Preprocessing identically matches the evaluation protocol:
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

### Grad-CAM Implementation
Targeting `model.layer4[-1].conv3`:
1. Register forward hook to capture activation tensor $A^k \in \mathbb{R}^{C \times H \times W}$.
2. Register backward hook to capture gradient tensor $\frac{\partial Y^c}{\partial A^k}$.
3. Calculate importance weights: $\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}$.
4. Compute weighted sum and apply ReLU: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
5. Bilinear interpolation to $224 \times 224$, apply JET colormap, and alpha-blend ($0.6 \times \text{Original} + 0.4 \times \text{Heatmap}$).

---

## 🚀 Setup & Execution (Windows / Cross-Platform)

### 1. Backend Setup

Open a terminal (PowerShell or Command Prompt):
```powershell
cd fetal-anomaly-ai\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# (Or Command Prompt: venv\Scripts\activate.bat)

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend will be available at:
- REST API: `http://localhost:8000`
- Interactive Swagger Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 2. Frontend Setup

Open a second terminal:
```powershell
cd fetal-anomaly-ai\frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at:
- Web Application: `http://localhost:3000`

---

## 📡 REST API Specifications

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns backend operational status, active device (`cuda`/`cpu`), and class count. |
| `GET` | `/model-info` | Returns model architecture metadata, target Grad-CAM layer, and normalization parameters. |
| `POST` | `/predict` | Multipart form data upload (`file`). Returns real prediction, confidence, probability distribution, and Grad-CAM image paths. |

---

## ⚖️ Medical & Research Disclaimer

> **IMPORTANT NOTICE:**  
> This application is an **AI research and educational prototype** developed exclusively for academic evaluation. Predictions generated by the system are **not medical diagnoses** and must **never** be used for clinical decision-making, triage, or patient management. All ultrasound examinations require assessment by certified medical imaging professionals.
