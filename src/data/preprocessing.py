"""
Image Preprocessing Module

Provides utilities for preprocessing chest X-ray images for deep learning models.
Includes normalization, resizing, and CXR-specific preprocessing functions.
"""

import logging
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image object
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def resize_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (512, 512),
    resample: int = Image.LANCZOS
) -> Union[Image.Image, np.ndarray]:
    """
    Resize image to target size.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target (width, height) or (height, width) tuple
        resample: Resampling filter (PIL.Image.LANCZOS, BILINEAR, etc.)
    
    Returns:
        Resized image (same type as input)
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image for resizing
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            # RGB
            pil_image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        pil_image = pil_image.resize(target_size, resample=resample)
        
        # Convert back to numpy array
        return np.array(pil_image)
    else:
        # PIL Image
        return image.resize(target_size, resample=resample)


def normalize_image(
    image: Union[Image.Image, np.ndarray],
    method: str = 'min_max',
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: PIL Image or numpy array
        method: Normalization method ('min_max', 'z_score', 'tanh')
        mean: Mean for z-score normalization (if None, computed from image)
        std: Standard deviation for z-score normalization (if None, computed from image)
    
    Returns:
        Normalized numpy array (float32, values in [0,1] for min_max, or standardized for z_score)
    
    Methods:
        - 'min_max': Scale to [0, 1]
        - 'z_score': Standardize (mean=0, std=1)
        - 'tanh': Scale to [-1, 1] using tanh
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image, dtype=np.float32)
    else:
        img_array = image.astype(np.float32)
    
    if method == 'min_max':
        # Scale to [0, 1]
        img_min = img_array.min()
        img_max = img_array.max()
        
        if img_max > img_min:
            normalized = (img_array - img_min) / (img_max - img_min)
        else:
            # Handle constant images
            normalized = np.zeros_like(img_array)
        
        return normalized
    
    elif method == 'z_score':
        # Standardize (mean=0, std=1)
        if mean is None:
            mean = img_array.mean()
        if std is None:
            std = img_array.std()
        
        if std > 0:
            normalized = (img_array - mean) / std
        else:
            # Handle constant images
            normalized = np.zeros_like(img_array)
        
        return normalized
    
    elif method == 'tanh':
        # Scale to [-1, 1] using tanh
        img_min = img_array.min()
        img_max = img_array.max()
        
        if img_max > img_min:
            # First normalize to [0, 1], then scale to [-1, 1]
            normalized = (img_array - img_min) / (img_max - img_min)
            normalized = 2 * normalized - 1
        else:
            normalized = np.zeros_like(img_array)
        
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose 'min_max', 'z_score', or 'tanh'")


def apply_window_level(
    image: Union[Image.Image, np.ndarray],
    window_center: float = 0.5,
    window_width: float = 1.0
) -> np.ndarray:
    """
    Apply window/level adjustment for CXR images (similar to DICOM windowing).
    
    This is useful for enhancing contrast in chest X-rays by focusing on a specific
    intensity range.
    
    Args:
        image: PIL Image or numpy array (should be normalized to [0, 1])
        window_center: Center of the window (0.0 to 1.0)
        window_width: Width of the window (0.0 to 1.0)
    
    Returns:
        Windowed numpy array (float32, values in [0, 1])
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image, dtype=np.float32)
    else:
        img_array = image.astype(np.float32)
    
    # Normalize to [0, 1] if not already
    if img_array.max() > 1.0:
        img_array = normalize_image(img_array, method='min_max')
    
    # Calculate window bounds
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # Clip values outside window
    windowed = np.clip(img_array, window_min, window_max)
    
    # Rescale to [0, 1]
    if window_max > window_min:
        windowed = (windowed - window_min) / (window_max - window_min)
    else:
        windowed = np.zeros_like(windowed)
    
    return windowed


def preprocess_image(
    image: Union[Image.Image, np.ndarray, Union[str, Path]],
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True,
    normalization_method: str = 'min_max',
    apply_window: bool = False,
    window_center: float = 0.5,
    window_width: float = 1.0,
    convert_to_grayscale: bool = True
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        target_size: Target (width, height) for resizing
        normalize: Whether to normalize the image
        normalization_method: Normalization method ('min_max', 'z_score', 'tanh')
        apply_window: Whether to apply window/level adjustment
        window_center: Window center for windowing (if apply_window=True)
        window_width: Window width for windowing (if apply_window=True)
        convert_to_grayscale: Convert RGB images to grayscale
    
    Returns:
        Preprocessed numpy array (float32, shape: (H, W) or (H, W, C))
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        image = load_image(image)
    
    # Convert to grayscale if needed
    if convert_to_grayscale and isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
    
    # Resize
    image = resize_image(image, target_size=target_size)
    
    # Convert to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image, dtype=np.float32)
    else:
        img_array = image.astype(np.float32)
    
    # Normalize
    if normalize:
        img_array = normalize_image(img_array, method=normalization_method)
    
    # Apply window/level adjustment
    if apply_window:
        img_array = apply_window_level(img_array, window_center, window_width)
    
    return img_array


def preprocess_mask(
    mask: Union[np.ndarray, Union[str, Path]],
    target_size: Tuple[int, int] = (512, 512),
    threshold: float = 0.5
) -> np.ndarray:
    """
    Preprocess segmentation mask.
    
    Args:
        mask: Numpy array or path to .npy mask file
        target_size: Target (width, height) for resizing
        threshold: Threshold for binarization (if mask is not already binary)
    
    Returns:
        Binary mask as numpy array (uint8, values 0 or 1)
    """
    # Load mask if path provided
    if isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        if mask_path.suffix == '.npy':
            mask = np.load(mask_path)
        else:
            # Try loading as image
            mask = np.array(Image.open(mask_path))
    
    # Ensure it's a numpy array
    mask = np.array(mask, dtype=np.float32)
    
    # Resize if needed
    if mask.shape[:2] != target_size[::-1]:  # target_size is (width, height), shape is (height, width)
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize(target_size, resample=Image.NEAREST)  # Use NEAREST for masks
        mask = np.array(mask_pil, dtype=np.float32)
    
    # Binarize if needed
    if mask.max() > 1.0 or mask.min() < 0.0:
        # Normalize first
        mask = normalize_image(mask, method='min_max')
    
    # Threshold to create binary mask
    binary_mask = (mask > threshold).astype(np.uint8)
    
    return binary_mask


def preprocess_image_mask_pair(
    image: Union[Image.Image, np.ndarray, Union[str, Path]],
    mask: Union[np.ndarray, Union[str, Path]],
    target_size: Tuple[int, int] = (512, 512),
    normalize: bool = True,
    normalization_method: str = 'min_max',
    apply_window: bool = False,
    window_center: float = 0.5,
    window_width: float = 1.0,
    convert_to_grayscale: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image and mask pair together (ensures same size).
    
    Args:
        image: PIL Image, numpy array, or path to image file
        mask: Numpy array or path to .npy mask file
        target_size: Target (width, height) for resizing
        normalize: Whether to normalize the image
        normalization_method: Normalization method
        apply_window: Whether to apply window/level adjustment
        window_center: Window center for windowing
        window_width: Window width for windowing
        convert_to_grayscale: Convert RGB images to grayscale
    
    Returns:
        Tuple of (preprocessed_image, preprocessed_mask) as numpy arrays
    """
    # Preprocess image
    preprocessed_image = preprocess_image(
        image=image,
        target_size=target_size,
        normalize=normalize,
        normalization_method=normalization_method,
        apply_window=apply_window,
        window_center=window_center,
        window_width=window_width,
        convert_to_grayscale=convert_to_grayscale
    )
    
    # Preprocess mask (use same target_size)
    preprocessed_mask = preprocess_mask(
        mask=mask,
        target_size=target_size
    )
    
    return preprocessed_image, preprocessed_mask
