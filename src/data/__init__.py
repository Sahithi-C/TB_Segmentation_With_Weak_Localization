"""
Data processing modules for TB segmentation project.
"""

from .weak_localization import (
    bbox_to_mask,
    generate_mask_for_image,
    generate_masks_for_dataset,
    load_mask,
    multiple_bboxes_to_mask,
    parse_bbox_string,
    validate_bbox,
    visualize_mask_overlay,
)

from .preprocessing import (
    load_image,
    resize_image,
    normalize_image,
    apply_window_level,
    preprocess_image,
    preprocess_mask,
    preprocess_image_mask_pair,
)

__all__ = [
    # Weak localization
    'parse_bbox_string',
    'validate_bbox',
    'bbox_to_mask',
    'multiple_bboxes_to_mask',
    'generate_mask_for_image',
    'generate_masks_for_dataset',
    'visualize_mask_overlay',
    'load_mask',
    # Preprocessing
    'load_image',
    'resize_image',
    'normalize_image',
    'apply_window_level',
    'preprocess_image',
    'preprocess_mask',
    'preprocess_image_mask_pair',
]
