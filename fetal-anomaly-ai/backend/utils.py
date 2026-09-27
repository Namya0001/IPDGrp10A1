"""
Utility functions for file validation, sanitization, and helpers.
"""

import os
import re
import uuid
import glob
from pathlib import Path
from PIL import Image
from io import BytesIO
from fastapi import UploadFile

from config import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    GRADCAM_OUTPUT_DIR,
)


class ValidationError(Exception):
    """Custom exception for file validation errors."""
    pass


async def validate_uploaded_file(file: UploadFile) -> bytes:
    """
    Validate an uploaded file for type, size, and integrity.

    Args:
        file: FastAPI UploadFile object

    Returns:
        bytes: The file content

    Raises:
        ValidationError: If validation fails
    """
    # Check filename exists
    if not file.filename:
        raise ValidationError("No filename provided.")

    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type: '{ext}'. "
            f"Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Check MIME type
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise ValidationError(
            f"Unsupported MIME type: '{file.content_type}'. "
            f"Supported types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File too large ({len(content) / (1024*1024):.1f} MB). "
            f"Maximum allowed size: {MAX_FILE_SIZE_MB} MB."
        )

    # Check file is not empty
    if len(content) == 0:
        raise ValidationError("Uploaded file is empty.")

    # Validate the image can be opened
    try:
        img = Image.open(BytesIO(content))
        img.verify()  # Verify it's a valid image
    except Exception:
        raise ValidationError(
            "The uploaded file appears to be corrupted or is not a valid image. "
            "Please upload a valid JPG, JPEG, or PNG image."
        )

    return content


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other security issues.

    Args:
        filename: Original filename

    Returns:
        str: Sanitized filename
    """
    # Get just the filename, not the path
    filename = Path(filename).name

    # Remove any non-alphanumeric characters except dots, hyphens, underscores
    filename = re.sub(r'[^\w\-.]', '_', filename)

    return filename


def generate_safe_filename(prefix: str = "upload", extension: str = ".png") -> str:
    """
    Generate a safe, unique filename.

    Args:
        prefix: Filename prefix
        extension: File extension

    Returns:
        str: Safe unique filename
    """
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{uid}{extension}"


def cleanup_old_gradcam_files(max_files: int = 100) -> None:
    """
    Clean up old Grad-CAM output files if too many accumulate.

    Args:
        max_files: Maximum number of files to keep
    """
    output_dir = GRADCAM_OUTPUT_DIR
    files = sorted(
        output_dir.glob("gradcam_*.png"),
        key=lambda f: f.stat().st_mtime
    )

    if len(files) > max_files:
        files_to_remove = files[:len(files) - max_files]
        for f in files_to_remove:
            try:
                f.unlink()
            except OSError:
                pass
