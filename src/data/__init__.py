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

__all__ = [
    'parse_bbox_string',
    'validate_bbox',
    'bbox_to_mask',
    'multiple_bboxes_to_mask',
    'generate_mask_for_image',
    'generate_masks_for_dataset',
    'visualize_mask_overlay',
    'load_mask',
]
