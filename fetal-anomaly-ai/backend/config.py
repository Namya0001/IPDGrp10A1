"""
Configuration module for the Fetal Anomaly Detection System backend.
All paths use pathlib for cross-platform compatibility.
"""

import os
from pathlib import Path

# Base directory (backend/)
BASE_DIR = Path(__file__).resolve().parent

# Model configuration
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_resnet50.pth")))
CLASS_MAPPING_PATH = Path(os.getenv("CLASS_MAPPING_PATH", str(BASE_DIR / "config" / "class_mapping.json")))

# Output directories
GRADCAM_OUTPUT_DIR = BASE_DIR / "outputs" / "gradcam"
GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inference configuration
INPUT_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Confidence threshold
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

# Upload constraints
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Model metadata
MODEL_NAME = "ResNet50"
TASK_NAME = "Fetal Ultrasound Anomaly Classification"
GRADCAM_LAYER = "layer4[-1].conv3"
