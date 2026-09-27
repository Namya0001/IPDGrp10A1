# Fetal Anomaly Detection System: Comprehensive Project Guide & Architecture Explanation

> **Final-Year Research Project:** *Early Detection of Fetal Anomalies During Pregnancy Using Deep Learning*  
> **Model:** Fine-tuned PyTorch ResNet50 (16 Anomaly Categories)  
> **Explainability:** Gradient-weighted Class Activation Mapping (Grad-CAM) targeting `layer4[-1].conv3`  
> **Full Stack:** Python FastAPI (REST API Backend) + Next.js 15 / React 19 / TypeScript / Tailwind CSS (Frontend)

---

## 1. Executive Summary & Research Context

Congenital fetal anomalies remain one of the primary causes of perinatal morbidity and mortality worldwide. Prenatal ultrasound (sonography) is the gold standard modality for assessing fetal anatomical development throughout gestational trimesters. However, the diagnostic accuracy of sonography depends heavily on operator expertise, acoustic shadowing, maternal body habitus, and subtle morphological variations in complex structures such as the fetal brain ventricles and posterior fossa.

This research project develops and deploys an end-to-end Computer Vision system capable of classifying fetal ultrasound scans into **16 distinct categories** (15 pathological conditions and 1 normal anatomical state) while providing visual explainability through **Grad-CAM**.

Importantly, this web application is powered by the **real trained PyTorch model (`best_resnet50.pth`)**. It does not use mocks, simulated predictions, hardcoded labels, or placeholder heatmaps. Every prediction, confidence value, and explainability overlay is computed in real-time by the neural network.

---

## 2. Dataset & Multi-Class Taxonomy

The model is trained to distinguish between **16 anomaly classes** as mapped in `class_mapping.json`:

| Index | Anomaly Class Name | Clinical Significance |
|:---:|---|---|
| **0** | `anold-chiari-malformation` | Arnold-Chiari malformation (structural defect in the cerebellum and skull base). |
| **1** | `arachnoid-cyst` | Benign fluid-filled sacs within the arachnoid membrane of the brain. |
| **2** | `cerebellah-hypoplasia` | Underdevelopment of the cerebellum, impacting motor control and neurodevelopment. |
| **3** | `colphocephaly` | Disproportionate enlargement of the occipital horns of the lateral ventricles. |
| **4** | `encephalocele` | Neural tube defect characterized by sac-like protrusions of the brain through skull defects. |
| **5** | `holoprosencephaly` | Failure of the prosencephalon (forebrain) to develop into two distinct hemispheres. |
| **6** | `hydracenphaly` | Condition in which the cerebral hemispheres are largely absent and replaced with cerebrospinal fluid (CSF). |
| **7** | `intracranial-hemorrdge` | Fetal intracranial bleeding within brain parenchyma or ventricular spaces. |
| **8** | `intracranial-tumor` | Space-occupying congenital neoplastic lesions within the fetal cranium. |
| **9** | `m-magna` | Mega cisterna magna (enlargement of the posterior fossa CSF space). |
| **10** | `mild-ventriculomegaly` | Dilation of fetal lateral ventricles (atrium diameter between 10.0 mm and 12.0 mm). |
| **11** | `moderate-ventriculomegaly` | Lateral ventricle diameter measuring between 12.1 mm and 14.9 mm. |
| **12** | `normal` | Normal gestational sonographic anatomy with no observable cranial/structural defect. |
| **13** | `polencephaly` | Porencephaly (cysts or cavities in a cerebral hemisphere, typically resulting from ischemic insult). |
| **14** | `severe-ventriculomegaly` | Severe ventricular enlargement (atrium diameter $\ge 15.0$ mm / hydrocephalus). |
| **15** | `vein-of-galen` | Vein of Galen aneurysmal malformation (high-flow arteriovenous fistula). |

### Real Model Performance Benchmarks (Holdout Test Set)
Extracted directly from the evaluation artifacts (`classification_report.csv`, `final_model_metrics.csv`):
- **Overall Test Accuracy:** `82.95%`
- **Macro-Averaged Precision:** `85.90%`
- **Macro-Averaged Recall:** `84.22%`
- **Macro-Averaged F1-Score:** `83.24%`
- **Weighted Multi-Class ROC-AUC:** `0.9732`

---

## 3. Deep Learning Architecture & Mathematical Pipeline

### 3.1 Network Architecture
The core model is based on **ResNet50** (Residual Network with 50 layers), utilizing residual skip connections to mitigate the vanishing gradient problem in deep neural architectures:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

1. **Feature Extractor:** 4 sequential residual blocks (`layer1` through `layer4`), containing bottleneck residual units with $1\times 1$, $3\times 3$, and $1\times 1$ convolutions.
2. **Classification Head:** The original 1000-class ImageNet fully connected layer is replaced with:
   $$\text{fc} = \text{nn.Linear}(\text{in\_features}=2048, \text{out\_features}=16)$$
3. **Weight Loading:** Pre-trained weights are loaded from `best_resnet50.pth` with weights frozen for inference (`model.eval()`).

### 3.2 Preprocessing Pipeline
To guarantee exact parity with training and validation, input images are processed deterministically:
1. **Format Harmonization:** Image converted to 3-channel RGB (ensures single-channel grayscale sonograms or 4-channel PNGs with alpha channels are converted cleanly).
2. **Bicubic/Bilinear Resizing:** Scaled to standard spatial dimensions: $224 \times 224$ pixels.
3. **Tensor Normalization:** Pixel values normalized using standard ImageNet parameters:
   $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}, \quad \boldsymbol{\mu} = [0.485, 0.456, 0.406], \quad \boldsymbol{\sigma} = [0.229, 0.224, 0.225]$$
4. **Batch Dimension Addition:** Transformed from shape `[3, 224, 224]` to `[1, 3, 224, 224]`.

### 3.3 Softmax Probability & Classification
Given raw network logits $\mathbf{z} = [z_0, z_1, \dots, z_{15}]$, the probability $p_i$ for each class $i$ is calculated using the Softmax function:

$$p_i = \frac{e^{z_i}}{\sum_{j=0}^{15} e^{z_j}}$$

The predicted category $c^*$ and confidence score $C$ are:

$$c^* = \arg\max_{i} (p_i), \quad C = \max_{i} (p_i)$$

---

## 4. Grad-CAM Explainability Mechanics

Because deep neural networks operate as non-linear feature extractors, clinicians require interpretability before considering AI-assisted insights. We implement **Grad-CAM (Gradient-weighted Class Activation Mapping)** targeting the final convolutional layer: `model.layer4[-1].conv3`.

```text
    Forward Pass: Input Tensor ──▶ ResNet50 Backbone ──▶ Activations A^k (layer4[-1].conv3) ──▶ Logits Y^c
                                                                                                  │
    Backward Pass: Target Score Y^c ──▶ Gradients d(Y^c)/d(A^k) ◀─────────────────────────────────┘
                                                  │
                                                  ▼
                                 Global Average Pooling (GAP)
                                                  │
                                                  ▼
                                 Neuron Importance Weights α_k^c
                                                  │
                                                  ▼
                                 Weighted Sum + ReLU Non-Linearity
                                                  │
                                                  ▼
                                    Grad-CAM Heatmap (224x224)
                                                  │
                                                  ▼
                                     JET Colormap + Alpha Blend
                                  (0.6 * Original + 0.4 * Heatmap)
```

### Mathematical Formulation:
1. **Activation Capture:** Forward hook captures feature maps $A^k \in \mathbb{R}^{U \times V}$ from `layer4[-1].conv3` ($k = 1, \dots, 2048$).
2. **Gradient Capture:** Backward hook captures the gradient of the predicted score $Y^c$ with respect to feature activation $A^k$:
   $$\frac{\partial Y^c}{\partial A_{i,j}^k}$$
3. **Channel Importance Weights:** Computed via Global Average Pooling over spatial dimensions (height $U$, width $V$):
   $$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
4. **Weighted Combination and Rectified Linear Unit (ReLU):**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
5. **Spatial Scaling and Visualization:**
   - Interpolated to $224 \times 224$ using OpenCV bilinear resizing.
   - Normalized to the interval $[0, 1]$.
   - Mapped to OpenCV `COLORMAP_JET` (blue = lowest influence, red = highest influence).
   - Alpha-blended with the original ultrasound scan: $\mathbf{I}_{\text{overlay}} = 0.6 \cdot \mathbf{I}_{\text{original}} + 0.4 \cdot \mathbf{I}_{\text{heatmap}}$.

---

## 5. End-to-End System Workflow

```text
[User Browser]
      │
      │ 1. Drag & Drop or Select Ultrasound Image (.png, .jpg, .jpeg)
      ▼
[Next.js 15 Client]
      │
      │ 2. Client-side validation: MIME type check, File size <= 10MB
      │ 3. Instant local image preview rendered via FileReader API
      │ 4. User clicks "Analyze Image" -> Triggers loading state
      ▼
[FastAPI REST API] (http://localhost:8000/predict)
      │
      │ 5. Server-side validation: Sanitization, PIL verification against corrupted images
      │ 6. Image transformed to RGB, resized to 224x224, normalized with ImageNet stats
      │ 7. PyTorch ResNet50 forward inference -> Softmax probabilities for all 16 classes
      │ 8. Forward & backward hooks compute Grad-CAM heatmap on layer4[-1].conv3
      │ 9. Heatmap and blended overlay saved to backend/outputs/gradcam/
      │ 10. API returns structured JSON with predictions, sorted probabilities & image paths
      ▼
[Next.js Results View]
      │
      │ 11. Prediction Card: Displays Top Class, Dynamic Colored Confidence Bar
      │ 12. Low Confidence Warning: Alert triggered if confidence < 0.60
      │ 13. Multi-Class Probability List: Horizontal ranked distribution across all 16 classes
      │ 14. Grad-CAM Explainability Viewer:
      │     - Tab switching: "Original", "Heatmap", "Overlay"
      │     - Real-time opacity slider (0% to 100%)
      │     - Side-by-side 3-panel comparison matrix
      │ 15. Medical Disclaimer: Emphasizes prototype research status
```

---

## 6. How to Run the Website (Step-by-Step)

The system consists of two independent services: the **FastAPI Python Backend** and the **Next.js TypeScript Frontend**.

### Prerequisites
- **Python:** 3.10, 3.11, 3.12, or 3.13 installed
- **Node.js:** 18.x, 20.x, or 22.x with `npm` installed
- **Operating System:** Windows 10/11, macOS, or Linux

---

### Step 1: Launch the Backend (FastAPI + PyTorch)

1. Open a new PowerShell / Command Prompt window:
   ```powershell
   cd c:\Users\ASUS\Desktop\ipd\fetal-anomaly-ai\backend
   ```

2. *(Optional but Recommended)* Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```

4. Start the FastAPI server using Uvicorn:
   ```powershell
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Verify Backend Health:**
   - Open your browser and navigate to: `http://localhost:8000/health`
   - You should see:
     ```json
     {
       "status": "ok",
       "model_loaded": true,
       "device": "cpu",
       "num_classes": 16
     }
     ```
   - Interactive Swagger API Documentation is accessible at: `http://localhost:8000/docs`

---

### Step 2: Launch the Frontend (Next.js 15)

1. Open a **second** PowerShell / Command Prompt window:
   ```powershell
   cd c:\Users\ASUS\Desktop\ipd\fetal-anomaly-ai\frontend
   ```

2. Install Node dependencies:
   ```powershell
   npm install
   ```

3. Start the Next.js development server:
   ```powershell
   npm run dev
   ```

4. **Access the Web Application:**
   - Open your browser and navigate to: `http://localhost:3000`

---

### Step 3: Performing an Ultrasound Analysis

1. From the homepage, click **"Analyze Ultrasound"** or navigate to the **Analyze** tab in the top navigation bar.
2. Drag and drop any fetal ultrasound image (or click **"Choose Image"** to browse your files). You can use any ultrasound scan from the project's `test/` directory.
3. The image preview and file details will appear immediately.
4. Click **"Analyze Image"**.
5. The loading sequence will indicate:
   - *Preprocessing image...*
   - *Running ResNet50 inference...*
   - *Generating Grad-CAM explanation...*
6. View the complete diagnostic analysis:
   - **Top Model Prediction** with category index.
   - **Confidence Percentage** with animated progress indicator.
   - **Interactive Grad-CAM Viewer** with tabs, opacity slider, and side-by-side view.
   - **Full Class Probability Breakdown** sorted from highest to lowest across all 16 classes.

---

## 7. Automated Backend Verification

You can run the built-in test suite anytime to verify that the model, class mapping, inference pipeline, and Grad-CAM generation are functioning properly:

```powershell
cd c:\Users\ASUS\Desktop\ipd\fetal-anomaly-ai\backend
python test_backend.py
```

Expected Output:
```text
============================================================
  RUNNING BACKEND VERIFICATION SUITE
============================================================
[Test 1] Checking Model & Config Files...
  [OK] Found model file (90.10 MB)
  [OK] Found class mapping file
[Test 2] Loading Model into memory...
  [OK] Model initialized with 16 classes on device: cpu
[Test 3] Testing Real Image Preprocessing & Inference...
  [OK] Preprocessed tensor shape: torch.Size([1, 3, 224, 224])
[Test 4] Executing Real ResNet50 Inference...
  [OK] Predicted Class: 'm-magna' (Index: 9)
  [OK] Confidence: 15.25%
  [OK] Probabilities sum: 1.0000 (Softmax normalized)
[Test 5] Generating Grad-CAM Heatmap & Overlay...
  [OK] Grad-CAM Heatmap generated: gradcam_heatmap_...png
  [OK] Grad-CAM Overlay generated: gradcam_overlay_...png
[Test 6] Testing FastAPI App Endpoints with TestClient...
  [OK] GET /health passed: status=ok, device=cpu, classes=16
  [OK] GET /model-info passed: model=ResNet50, gradcam_layer=layer4[-1].conv3
  [OK] POST /predict passed: Predicted with confidence
============================================================
  ALL BACKEND TESTS PASSED SUCCESSFULLY! (100% REAL MODEL)
============================================================
```

---

## 8. Academic & Medical Advisory

> **Research Prototype Statement:**  
> This software is developed strictly as an academic research and educational demonstration platform. Model classifications and Grad-CAM heatmaps do **not** constitute medical diagnoses or clinical findings. Ultrasound interpretation requires clinical correlation by board-certified radiologists, perinatologists, and obstetric sonologists.
