"""
Automated validation and testing script for Fetal Anomaly Detection backend.
Verifies:
1. Model loading with real weights best_resnet50.pth
2. Class mapping loading (16 classes)
3. Preprocessing pipeline
4. Real inference pass (no mock/fake output)
5. Grad-CAM generation with layer4[-1].conv3 hooks
6. FastAPI endpoints (/health, /model-info, /predict)
"""

import sys
import glob
from pathlib import Path
from PIL import Image
import torch

# Ensure backend directory is in path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from model import model_instance
from preprocessing import preprocess_image
from gradcam import generate_gradcam_images
from config import MODEL_PATH, CLASS_MAPPING_PATH


def run_tests():
    print("=" * 60)
    print("  RUNNING BACKEND VERIFICATION SUITE")
    print("=" * 60)

    # 1. Verify Model Files
    print("\n[Test 1] Checking Model & Config Files...")
    assert MODEL_PATH.exists(), f"Missing model at {MODEL_PATH}"
    assert CLASS_MAPPING_PATH.exists(), f"Missing class mapping at {CLASS_MAPPING_PATH}"
    print(f"  [OK] Found model file ({MODEL_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  [OK] Found class mapping file")

    # 2. Load Model
    print("\n[Test 2] Loading Model into memory...")
    model_instance.load()
    assert model_instance.is_loaded, "Model failed to load"
    assert model_instance.num_classes == 16, f"Expected 16 classes, got {model_instance.num_classes}"
    print(f"  [OK] Model initialized with {model_instance.num_classes} classes on device: {model_instance.device}")

    # 3. Find a test image from test directory or generate a synthetic ultrasound-like image
    print("\n[Test 3] Testing Real Image Preprocessing & Inference...")
    sample_images = list(Path("c:/Users/ASUS/Desktop/ipd/test").rglob("*.jpg")) + list(Path("c:/Users/ASUS/Desktop/ipd/test").rglob("*.png"))
    
    if sample_images:
        test_img_path = sample_images[0]
        print(f"  Using real test sample: {test_img_path.name}")
        img = Image.open(test_img_path)
    else:
        print("  Creating test RGB image...")
        img = Image.new("RGB", (300, 300), color=(100, 100, 100))

    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224), f"Unexpected tensor shape {tensor.shape}"
    print(f"  [OK] Preprocessed tensor shape: {tensor.shape}")

    # 4. Inference
    print("\n[Test 4] Executing Real ResNet50 Inference...")
    pred_label, pred_idx, confidence, all_probs = model_instance.predict(tensor)
    print(f"  [OK] Predicted Class: '{pred_label}' (Index: {pred_idx})")
    print(f"  [OK] Confidence: {confidence * 100:.2f}%")
    print(f"  [OK] Probabilities sum: {sum(all_probs.values()):.4f} (Softmax normalized)")
    assert len(all_probs) == 16, f"Expected 16 class probabilities, got {len(all_probs)}"

    # 5. Grad-CAM Generation
    print("\n[Test 5] Generating Grad-CAM Heatmap & Overlay...")
    heatmap_fn, overlay_fn = generate_gradcam_images(
        model=model_instance.model,
        input_tensor=tensor,
        original_image=img,
        predicted_class_idx=pred_idx,
        device=model_instance.device
    )
    heatmap_file = BACKEND_DIR / "outputs" / "gradcam" / heatmap_fn
    overlay_file = BACKEND_DIR / "outputs" / "gradcam" / overlay_fn
    assert heatmap_file.exists(), f"Missing generated heatmap: {heatmap_file}"
    assert overlay_file.exists(), f"Missing generated overlay: {overlay_file}"
    print(f"  [OK] Grad-CAM Heatmap generated: {heatmap_file.name}")
    print(f"  [OK] Grad-CAM Overlay generated: {overlay_file.name}")

    # 6. Test FastAPI test client
    print("\n[Test 6] Testing FastAPI App Endpoints with TestClient...")
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    
    # Health endpoint
    h_resp = client.get("/health")
    assert h_resp.status_code == 200, f"Health returned {h_resp.status_code}"
    h_data = h_resp.json()
    assert h_data["status"] == "ok"
    assert h_data["model_loaded"] is True
    assert h_data["num_classes"] == 16
    print(f"  [OK] GET /health passed: status={h_data['status']}, device={h_data['device']}, classes={h_data['num_classes']}")

    # Model info endpoint
    m_resp = client.get("/model-info")
    assert m_resp.status_code == 200, f"Model info returned {m_resp.status_code}"
    m_data = m_resp.json()
    assert m_data["model"] == "ResNet50"
    assert m_data["num_classes"] == 16
    print(f"  [OK] GET /model-info passed: model={m_data['model']}, gradcam_layer={m_data['gradcam_layer']}")

    # Predict endpoint
    import io
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    p_resp = client.post(
        "/predict",
        files={"file": ("sample_ultrasound.png", img_byte_arr, "image/png")}
    )
    assert p_resp.status_code == 200, f"Predict returned {p_resp.status_code}: {p_resp.text}"
    p_data = p_resp.json()
    assert p_data["success"] is True
    assert "prediction" in p_data
    assert "gradcam" in p_data
    assert len(p_data["probabilities"]) == 16
    print(f"  [OK] POST /predict passed: Predicted '{p_data['prediction']['class_name']}' with confidence {p_data['prediction']['confidence']*100:.2f}%")

    print("\n" + "=" * 60)
    print("  ALL BACKEND TESTS PASSED SUCCESSFULLY! (100% REAL MODEL)")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
