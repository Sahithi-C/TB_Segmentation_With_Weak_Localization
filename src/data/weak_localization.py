"""
Weak Localization Module

Converts bounding box annotations to segmentation masks for weak supervision training.
This module handles the conversion of TBX11K bounding box annotations into pixel-level masks
that can be used for training U-Net segmentation models.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_bbox_string(bbox_str: Union[str, Dict]) -> Optional[Dict]:
    """
    Parse bounding box string from CSV into dictionary.
    
    TBX11K stores bounding boxes as Python dictionary strings (single quotes),
    not JSON (double quotes), so we use ast.literal_eval() instead of json.loads().
    
    Args:
        bbox_str: Bounding box string from CSV (e.g., "{'xmin': 100, 'ymin': 200, 'width': 50, 'height': 60}")
                  or 'none' for no bounding box, or already a dictionary.
    
    Returns:
        Dictionary with keys: 'xmin', 'ymin', 'width', 'height', or None if invalid/no bbox.
    
    Example:
        >>> bbox_str = "{'xmin': 381.83, 'ymin': 126.87, 'width': 40.24, 'height': 44.57}"
        >>> parse_bbox_string(bbox_str)
        {'xmin': 381.83, 'ymin': 126.87, 'width': 40.24, 'height': 44.57}
    """
    if bbox_str is None or bbox_str == 'none' or pd.isna(bbox_str):
        return None
    
    if isinstance(bbox_str, dict):
        return bbox_str
    
    if not isinstance(bbox_str, str):
        logger.warning(f"Unexpected bbox type: {type(bbox_str)}")
        return None
    
    try:
        bbox_dict = ast.literal_eval(bbox_str)
        if not isinstance(bbox_dict, dict):
            logger.warning(f"Parsed bbox is not a dict: {type(bbox_dict)}")
            return None
        return bbox_dict
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse bbox string '{bbox_str}': {e}")
        return None


def validate_bbox(bbox: Dict, image_shape: Tuple[int, int]) -> bool:
    """
    Validate bounding box coordinates are within image bounds.
    
    Args:
        bbox: Dictionary with 'xmin', 'ymin', 'width', 'height'
        image_shape: (height, width) of the image
    
    Returns:
        True if bbox is valid, False otherwise.
    """
    if bbox is None:
        return False
    
    required_keys = ['xmin', 'ymin', 'width', 'height']
    if not all(key in bbox for key in required_keys):
        return False
    
    h, w = image_shape
    xmin = bbox['xmin']
    ymin = bbox['ymin']
    width = bbox['width']
    height = bbox['height']
    
    # Check for valid values
    if width <= 0 or height <= 0:
        return False
    
    # Calculate max coordinates
    xmax = xmin + width
    ymax = ymin + height
    
    # Check bounds (allow slight overflow for float coordinates)
    if xmin < 0 or ymin < 0 or xmax > w + 1 or ymax > h + 1:
        return False
    
    return True


def bbox_to_mask(bbox: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a single bounding box to a binary mask.
    
    Args:
        bbox: Dictionary with 'xmin', 'ymin', 'width', 'height'
        image_shape: (height, width) of the image
    
    Returns:
        Binary mask as numpy array of shape (height, width) with dtype uint8.
        Pixels inside bbox are 1, outside are 0.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if bbox is None:
        return mask
    
    xmin = int(bbox['xmin'])
    ymin = int(bbox['ymin'])
    width = int(bbox['width'])
    height = int(bbox['height'])
    
    # Calculate bounds (clip to image dimensions)
    xmax = min(xmin + width, w)
    ymax = min(ymin + height, h)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    
    # Set mask region to 1
    mask[ymin:ymax, xmin:xmax] = 1
    
    return mask


def multiple_bboxes_to_mask(bboxes: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert multiple bounding boxes to a single binary mask.
    
    If multiple boxes overlap, the union is taken (pixels in any box = 1).
    
    Args:
        bboxes: List of bounding box dictionaries
        image_shape: (height, width) of the image
    
    Returns:
        Binary mask as numpy array of shape (height, width) with dtype uint8.
        Pixels inside any bbox are 1, outside all boxes are 0.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not bboxes:
        return mask
    
    for bbox in bboxes:
        if bbox is None:
            continue
        
        if not validate_bbox(bbox, image_shape):
            logger.warning(f"Invalid bbox skipped: {bbox}")
            continue
        
        # Create mask for this bbox
        bbox_mask = bbox_to_mask(bbox, image_shape)
        # Union with existing mask
        mask = np.maximum(mask, bbox_mask)
    
    return mask


def generate_mask_for_image(
    image_path: Union[str, Path],
    bboxes: List[Union[str, Dict]],
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Generate segmentation mask for a single image from its bounding boxes.
    
    Args:
        image_path: Path to the image file
        bboxes: List of bounding box strings or dictionaries (from CSV rows)
        output_path: Optional path to save the mask. If None, mask is not saved.
    
    Returns:
        Binary mask as numpy array of shape (image_height, image_width).
    """
    # Load image to get dimensions
    try:
        img = Image.open(image_path)
        image_shape = (img.height, img.width)
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise
    
    # Parse bounding boxes
    parsed_bboxes = []
    for bbox in bboxes:
        parsed = parse_bbox_string(bbox)
        if parsed is not None:
            parsed_bboxes.append(parsed)
    
    # Generate mask
    mask = multiple_bboxes_to_mask(parsed_bboxes, image_shape)
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as numpy array (.npy) for efficient loading
        np.save(output_path, mask)
        logger.debug(f"Saved mask to {output_path}")
    
    return mask


def generate_masks_for_dataset(
    csv_path: Union[str, Path],
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_col: str = 'fname',
    bbox_col: str = 'bbox',
    image_height_col: str = 'image_height',
    image_width_col: str = 'image_width'
) -> Dict[str, int]:
    """
    Generate masks for all images in the dataset.
    
    Handles multiple bounding boxes per image (multiple CSV rows per filename).
    Creates one mask per unique image filename.
    
    Args:
        csv_path: Path to CSV file with bounding box annotations
        images_dir: Directory containing image files
        output_dir: Directory to save generated masks
        image_col: Column name for image filename
        bbox_col: Column name for bounding box
        image_height_col: Column name for image height (for validation)
        image_width_col: Column name for image width (for validation)
    
    Returns:
        Dictionary with statistics:
        - 'total_images': Number of unique images processed
        - 'masks_generated': Number of masks successfully generated
        - 'images_with_bbox': Number of images with at least one bounding box
        - 'images_without_bbox': Number of images with no bounding boxes
        - 'errors': Number of errors encountered
    """
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Group by image filename (handle multiple rows per image)
    logger.info("Grouping rows by image filename...")
    grouped = df.groupby(image_col)
    
    stats = {
        'total_images': len(grouped),
        'masks_generated': 0,
        'images_with_bbox': 0,
        'images_without_bbox': 0,
        'errors': 0
    }
    
    # Process each image
    logger.info(f"Generating masks for {stats['total_images']} images...")
    for fname, group_df in tqdm(grouped, desc="Generating masks"):
        try:
            # Get all bounding boxes for this image
            bboxes = group_df[bbox_col].tolist()
            
            # Check if any bbox exists
            has_bbox = any(
                bbox is not None and bbox != 'none' and not pd.isna(bbox)
                for bbox in bboxes
            )
            
            # Image path
            image_path = images_dir / fname
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                stats['errors'] += 1
                continue
            
            # Generate mask
            mask_output_path = output_dir / f"{Path(fname).stem}.npy"
            mask = generate_mask_for_image(image_path, bboxes, mask_output_path)
            
            # Update statistics
            stats['masks_generated'] += 1
            if has_bbox:
                stats['images_with_bbox'] += 1
            else:
                stats['images_without_bbox'] += 1
                
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")
            stats['errors'] += 1
            continue
    
    logger.info(f"Mask generation complete!")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Masks generated: {stats['masks_generated']}")
    logger.info(f"  Images with bbox: {stats['images_with_bbox']}")
    logger.info(f"  Images without bbox: {stats['images_without_bbox']}")
    logger.info(f"  Errors: {stats['errors']}")
    
    return stats


def visualize_mask_overlay(
    image_path: Union[str, Path],
    mask: np.ndarray,
    bboxes: Optional[List[Dict]] = None,
    save_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.5
) -> None:
    """
    Visualize image with mask overlay and optional bounding box rectangles.
    
    Args:
        image_path: Path to the image file
        mask: Binary mask array
        bboxes: Optional list of bounding box dictionaries to draw rectangles
        save_path: Optional path to save the visualization
        alpha: Transparency of mask overlay (0.0 to 1.0)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        logger.error("matplotlib not available for visualization")
        return
    
    # Load image
    img = Image.open(image_path)
    if img.mode == 'RGB':
        img_array = np.array(img)
    else:
        img_array = np.array(img.convert('RGB'))
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display image
    ax.imshow(img_array, cmap='gray' if img.mode != 'RGB' else None)
    
    # Overlay mask (red overlay)
    mask_colored = np.zeros((*mask.shape, 4), dtype=np.float32)
    mask_colored[:, :, 0] = 1.0  # Red channel
    mask_colored[:, :, 3] = mask.astype(np.float32) * alpha  # Alpha channel
    ax.imshow(mask_colored)
    
    # Draw bounding boxes if provided
    if bboxes:
        for bbox in bboxes:
            if bbox is None:
                continue
            xmin = bbox.get('xmin', 0)
            ymin = bbox.get('ymin', 0)
            width = bbox.get('width', 0)
            height = bbox.get('height', 0)
            
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            ax.add_patch(rect)
    
    ax.set_title(f"Image: {Path(image_path).name}\nMask Overlay (Red) + BBoxes (Yellow)")
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved visualization to {save_path}")
    
    # Show the plot (for Colab/Jupyter notebooks)
    plt.show()
    # Note: Don't close the figure here - let the caller handle it
    # This ensures the figure displays properly in Colab


def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """
    Load a saved mask from .npy file.
    
    Args:
        mask_path: Path to .npy mask file
    
    Returns:
        Binary mask as numpy array
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask = np.load(mask_path)
    return mask.astype(np.uint8)
