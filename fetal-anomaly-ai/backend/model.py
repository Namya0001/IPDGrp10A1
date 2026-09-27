"""
Model loading and inference module.
Loads the trained ResNet50 model once at startup and provides inference functions.
"""

import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
from pathlib import Path
from typing import Dict, Tuple, Optional

from config import MODEL_PATH, CLASS_MAPPING_PATH


class FetalAnomalyModel:
    """
    Singleton-style wrapper for the trained ResNet50 fetal anomaly classifier.
    Loads the model and class mapping once, provides inference methods.
    """

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.class_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_class: Optional[Dict[int, str]] = None
        self.num_classes: int = 0
        self.is_loaded: bool = False

    def load(self) -> None:
        """
        Load the class mapping and trained model weights.
        Called once at application startup.
        """
        # Validate files exist
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                f"Please ensure best_resnet50.pth is placed in {MODEL_PATH.parent}"
            )

        if not CLASS_MAPPING_PATH.exists():
            raise FileNotFoundError(
                f"Class mapping file not found: {CLASS_MAPPING_PATH}\n"
                f"Please ensure class_mapping.json is placed in {CLASS_MAPPING_PATH.parent}"
            )

        # Load class mapping
        with open(CLASS_MAPPING_PATH, "r") as f:
            self.class_to_idx = json.load(f)

        self.idx_to_class = {
            int(index): class_name.strip()
            for class_name, index in self.class_to_idx.items()
        }
        self.num_classes = len(self.class_to_idx)

        print(f"[Model] Loaded class mapping: {self.num_classes} classes")
        for idx, name in sorted(self.idx_to_class.items()):
            print(f"  [{idx}] {name}")

        # Detect device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Model] Using device: {self.device}")

        # Create ResNet50 architecture with correct number of output classes
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.num_classes
        )

        # Load trained weights
        state_dict = torch.load(
            str(MODEL_PATH),
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(state_dict)

        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        self.is_loaded = True
        print(f"[Model] ResNet50 loaded successfully from {MODEL_PATH}")
        print(f"[Model] Model is on device: {self.device}")

    def predict(self, input_tensor: torch.Tensor) -> Tuple[str, int, float, Dict[str, float]]:
        """
        Run inference on a preprocessed input tensor.

        Args:
            input_tensor: Preprocessed tensor of shape [1, 3, 224, 224]

        Returns:
            Tuple of (predicted_label, predicted_index, confidence, all_probabilities)
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # Move input to device
        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Calculate softmax probabilities
        probabilities = torch.softmax(output, dim=1)

        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        predicted_label = self.idx_to_class[predicted_class]

        # Get all class probabilities
        all_probs = {}
        for idx in range(self.num_classes):
            class_name = self.idx_to_class[idx]
            prob = probabilities[0, idx].item()
            all_probs[class_name] = prob

        return predicted_label, predicted_class, confidence, all_probs


# Global singleton instance
model_instance = FetalAnomalyModel()
