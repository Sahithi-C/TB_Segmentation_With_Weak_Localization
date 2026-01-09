"""
Helper utilities for TB Segmentation project.

Provides common utility functions for file operations, path management, etc.
"""

import os
import json
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

# Optional imports for functions that require them
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Assuming this file is in src/utils/, go up to project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    return project_root


def save_json(data: Dict[Any, Any], filepath: Union[str, Path], indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[Any, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    if torch is None:
        raise ImportError("PyTorch is required for count_parameters function")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device object or string
    """
    if torch is None:
        return "cpu"  # Return string if PyTorch not available
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    
    if np is not None:
        np.random.seed(seed)
    
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]):
    """
    Find the latest checkpoint file in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]


def create_experiment_dir(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = base_dir / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def print_model_summary(model, input_size: tuple = (1, 1, 512, 512)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    if torch is None:
        raise ImportError("PyTorch is required for print_model_summary function")
    
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print("=" * 80)
    
    # Try to print model structure (if torchsummary is available)
    try:
        from torchsummary import summary
        summary(model, input_size[1:])
    except ImportError:
        print("Install torchsummary for detailed model summary: pip install torchsummary")
        print("\nModel architecture:")
        print(model)
    except Exception as e:
        print(f"Could not generate detailed summary: {e}")
        print("\nModel architecture:")
        print(model)
    
    print("=" * 80)

