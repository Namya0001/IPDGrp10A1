"""
FastAPI application for Fetal Anomaly Detection System.
Provides REST API endpoints for image classification using a trained ResNet50 model.
"""

import traceback
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import (
    CORS_ORIGINS,
    CONFIDENCE_THRESHOLD,
    MODEL_NAME,
    TASK_NAME,
    GRADCAM_LAYER,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    GRADCAM_OUTPUT_DIR,
)
from model import model_instance
from preprocessing import preprocess_image
from gradcam import generate_gradcam_images
from schemas import (
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
    ClassProbability,
    PredictionResult,
    GradCAMResult,
    ErrorResponse,
)
from utils import validate_uploaded_file, ValidationError, cleanup_old_gradcam_files


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, cleanup at shutdown."""
    print("=" * 60)
    print("  Fetal Anomaly Detection System — Starting Up")
    print("=" * 60)

    try:
        model_instance.load()
        print("\n[Startup] Model loaded successfully!")
        print(f"[Startup] Classes: {model_instance.num_classes}")
        print(f"[Startup] Device: {model_instance.device}")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("[ERROR] The application will start but predictions will fail.")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        print("=" * 60)

    yield

    # Shutdown cleanup
    print("\n[Shutdown] Fetal Anomaly Detection System shutting down.")


# Create FastAPI app
app = FastAPI(
    title="Fetal Anomaly Detection System",
    description=(
        "AI-assisted analysis of fetal ultrasound images using a trained "
        "ResNet50 deep learning model with Grad-CAM explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Grad-CAM output images as static files
app.mount(
    "/outputs/gradcam",
    StaticFiles(directory=str(GRADCAM_OUTPUT_DIR)),
    name="gradcam_outputs",
)


# ─── Health Check ───────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are operational."""
    return HealthResponse(
        status="ok" if model_instance.is_loaded else "model_not_loaded",
        model_loaded=model_instance.is_loaded,
        device=str(model_instance.device) if model_instance.device else "unknown",
        num_classes=model_instance.num_classes,
    )


# ─── Model Info ─────────────────────────────────────────────


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model configuration and metadata."""
    return ModelInfoResponse(
        model=MODEL_NAME,
        task=TASK_NAME,
        input_size=f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        num_classes=model_instance.num_classes,
        gradcam_layer=GRADCAM_LAYER,
        preprocessing={
            "resize": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
        },
    )


# ─── Predict ────────────────────────────────────────────────


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze a fetal ultrasound image using the trained ResNet50 model.

    Accepts a multipart/form-data upload with an image file.
    Returns the predicted class, confidence, all class probabilities,
    and Grad-CAM explainability images.
    """
    # Check model is loaded
    if not model_instance.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check the server logs.",
        )

    # Validate uploaded file
    try:
        content = await validate_uploaded_file(file)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Open image
    try:
        original_image = Image.open(BytesIO(content))
        if original_image.mode != "RGB":
            original_image = original_image.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to open the uploaded image. Please verify it is a valid JPG, JPEG, or PNG file.",
        )

    # Preprocess
    try:
        input_tensor = preprocess_image(original_image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to preprocess the image.",
        )

    # Run inference
    try:
        predicted_label, predicted_idx, confidence, all_probs = model_instance.predict(input_tensor)
    except Exception as e:
        print(f"[Predict Error] Inference failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again.",
        )

    # Generate Grad-CAM
    try:
        heatmap_filename, overlay_filename = generate_gradcam_images(
            model=model_instance.model,
            input_tensor=input_tensor,
            original_image=original_image,
            predicted_class_idx=predicted_idx,
            device=model_instance.device,
        )
    except Exception as e:
        print(f"[GradCAM Error] {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Failed to generate Grad-CAM explanation.",
        )

    # Clean up old files periodically
    try:
        cleanup_old_gradcam_files(max_files=200)
    except Exception:
        pass  # Non-critical

    # Build sorted probabilities list (highest first)
    prob_list = [
        ClassProbability(
            class_name=class_name,
            class_index=model_instance.class_to_idx.get(
                class_name,
                model_instance.class_to_idx.get(f" {class_name}", -1)
            ),
            probability=round(prob, 6),
        )
        for class_name, prob in all_probs.items()
    ]
    # Fix class_index using idx_to_class reverse lookup
    prob_list_fixed = []
    for class_name, prob in all_probs.items():
        # Find the index for this class_name in idx_to_class
        class_index = -1
        for idx, name in model_instance.idx_to_class.items():
            if name == class_name:
                class_index = idx
                break
        prob_list_fixed.append(
            ClassProbability(
                class_name=class_name,
                class_index=class_index,
                probability=round(prob, 6),
            )
        )

    prob_list_fixed.sort(key=lambda x: x.probability, reverse=True)

    # Determine low confidence
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    return PredictResponse(
        success=True,
        prediction=PredictionResult(
            class_name=predicted_label,
            class_index=predicted_idx,
            confidence=round(confidence, 6),
        ),
        probabilities=prob_list_fixed,
        gradcam=GradCAMResult(
            heatmap_url=f"/outputs/gradcam/{heatmap_filename}",
            overlay_url=f"/outputs/gradcam/{overlay_filename}",
        ),
        low_confidence=low_confidence,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
