"""
Image preprocessing module.
Exactly matches the preprocessing used during training/evaluation.
"""

from torchvision import transforms
from PIL import Image
from config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_inference_transform():
    """
    Returns the inference preprocessing pipeline.
    Matches the evaluation notebook exactly:
    Resize(224,224) -> ToTensor -> Normalize(ImageNet mean/std)
    """
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def preprocess_image(image: Image.Image):
    """
    Preprocesses a PIL Image for model inference.

    Steps:
    1. Convert to RGB
    2. Apply inference transform (Resize, ToTensor, Normalize)
    3. Add batch dimension

    Args:
        image: PIL Image object

    Returns:
        torch.Tensor: Preprocessed tensor with shape [1, 3, 224, 224]
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_inference_transform()
    tensor = transform(image)

    # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    tensor = tensor.unsqueeze(0)

    return tensor
