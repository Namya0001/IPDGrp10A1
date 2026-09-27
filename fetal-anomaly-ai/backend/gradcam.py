"""
Grad-CAM implementation for ResNet50.
Generates visual explanations by highlighting regions that contributed to the model's prediction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple
import uuid

from config import GRADCAM_OUTPUT_DIR, INPUT_SIZE


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Targets model.layer4[-1].conv3 for ResNet50.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: The ResNet50 model
            target_layer: The convolutional layer to generate Grad-CAM for
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed input tensor [1, 3, 224, 224]
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            np.ndarray: Normalized heatmap of shape (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Enable gradients for Grad-CAM
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # If no class specified, use predicted class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Select the score for the target class
        target_score = output[0, class_idx]

        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=False)

        # Get activations and gradients
        activations = self.activations  # [1, C, H, W]
        gradients = self.gradients  # [1, C, H, W]

        # Global average pooling of gradients -> channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU
        cam = F.relu(cam)

        # Remove batch and channel dimensions
        cam = cam.squeeze().cpu().numpy()  # [H, W]

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def generate_gradcam_images(
    model,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    predicted_class_idx: int,
    device: torch.device
) -> Tuple[str, str]:
    """
    Generate Grad-CAM heatmap and overlay images.

    Args:
        model: The loaded ResNet50 model
        input_tensor: Preprocessed input tensor
        original_image: Original PIL Image (before preprocessing)
        predicted_class_idx: The predicted class index
        device: torch device

    Returns:
        Tuple of (heatmap_filename, overlay_filename)
    """
    # Target layer: model.layer4[-1].conv3
    target_layer = model.layer4[-1].conv3

    # Create Grad-CAM instance
    gradcam = GradCAM(model, target_layer)

    # Need gradients, so use a fresh tensor
    input_for_gradcam = input_tensor.clone().to(device)

    # Generate heatmap
    heatmap = gradcam.generate(input_for_gradcam, class_idx=predicted_class_idx)

    # Resize original image for overlay
    original_resized = original_image.copy()
    if original_resized.mode != "RGB":
        original_resized = original_resized.convert("RGB")
    original_resized = original_resized.resize(INPUT_SIZE, Image.LANCZOS)
    original_np = np.array(original_resized)  # [H, W, 3] in RGB

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, INPUT_SIZE)

    # Generate unique filenames
    uid = uuid.uuid4().hex[:12]
    heatmap_filename = f"gradcam_heatmap_{uid}.png"
    overlay_filename = f"gradcam_overlay_{uid}.png"

    # Save heatmap image (colorized)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    # Convert BGR to RGB for saving with PIL
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(heatmap_colored_rgb)
    heatmap_path = GRADCAM_OUTPUT_DIR / heatmap_filename
    heatmap_img.save(str(heatmap_path))

    # Create overlay: blend original with heatmap
    overlay = cv2.addWeighted(
        original_np,
        0.6,
        heatmap_colored_rgb,
        0.4,
        0
    )
    overlay_img = Image.fromarray(overlay)
    overlay_path = GRADCAM_OUTPUT_DIR / overlay_filename
    overlay_img.save(str(overlay_path))

    return heatmap_filename, overlay_filename
