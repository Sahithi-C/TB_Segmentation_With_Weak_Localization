"""
Configuration management system for TB Segmentation project.

This module provides a YAML-based configuration loader with validation.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "unet_modality_specific"
    encoder: str = "resnet34"
    pretrained: bool = True
    num_classes: int = 1
    in_channels: int = 1
    features: int = 64


@dataclass
class DataConfig:
    """Data configuration parameters."""
    image_size: list = field(default_factory=lambda: [512, 512])
    batch_size: int = 16
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    data_root: str = "data"
    datasets: list = field(default_factory=lambda: ["tbx11k", "shenzhen", "montgomery"])


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: str = "cosine"
    loss: str = "dice_iou_combined"
    early_stopping_patience: int = 10
    save_dir: str = "results/models"
    log_dir: str = "experiments/training"


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    rotation: int = 15
    horizontal_flip: bool = True
    vertical_flip: bool = False
    contrast_range: list = field(default_factory=lambda: [0.8, 1.2])
    brightness_range: list = field(default_factory=lambda: [0.8, 1.2])
    elastic_transform: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""
    metrics: list = field(default_factory=lambda: ["dice", "iou", "sensitivity", "specificity"])
    save_predictions: bool = True
    generate_visualizations: bool = True
    output_dir: str = "results"


class Config:
    """
    Main configuration class for the TB Segmentation project.
    
    Loads configuration from YAML files and provides access to nested configs.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Dictionary with configuration (alternative to file)
        """
        if config_dict:
            self._config = config_dict
        elif config_path:
            self._config = self._load_from_yaml(config_path)
        else:
            self._config = self._get_default_config()
        
        # Validate configuration
        self.validate()
        
        # Create config objects
        self.model = ModelConfig(**self._config.get("model", {}))
        self.data = DataConfig(**self._config.get("data", {}))
        self.training = TrainingConfig(**self._config.get("training", {}))
        self.augmentation = AugmentationConfig(**self._config.get("augmentation", {}))
        self.evaluation = EvaluationConfig(**self._config.get("evaluation", {}))
    
    def _load_from_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigError(f"Error parsing YAML file: {e}")
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "name": "unet_modality_specific",
                "encoder": "resnet34",
                "pretrained": True,
                "num_classes": 1
            },
            "data": {
                "image_size": [512, 512],
                "batch_size": 16,
                "num_workers": 4,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "scheduler": "cosine",
                "loss": "dice_iou_combined"
            }
        }
    
    def validate(self):
        """Validate configuration parameters."""
        # Validate data splits
        if "data" in self._config:
            data_config = self._config["data"]
            train_split = data_config.get("train_split", 0.7)
            val_split = data_config.get("val_split", 0.15)
            test_split = data_config.get("test_split", 0.15)
            
            total = train_split + val_split + test_split
            if abs(total - 1.0) > 0.01:
                raise ConfigError(
                    f"Data splits must sum to 1.0, got {total}: "
                    f"train={train_split}, val={val_split}, test={test_split}"
                )
        
        # Validate image size
        if "data" in self._config:
            image_size = self._config["data"].get("image_size", [512, 512])
            if len(image_size) != 2:
                raise ConfigError(f"image_size must be [height, width], got {image_size}")
        
        # Validate learning rate
        if "training" in self._config:
            lr = self._config["training"].get("learning_rate", 0.001)
            if lr <= 0:
                raise ConfigError(f"learning_rate must be positive, got {lr}")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.data
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        return self.training
    
    def get_augmentation_config(self) -> AugmentationConfig:
        """Get augmentation configuration."""
        return self.augmentation
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self.evaluation
    
    def save(self, output_path: str):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, updates)
        self.validate()
        
        # Recreate config objects
        self.model = ModelConfig(**self._config.get("model", {}))
        self.data = DataConfig(**self._config.get("data", {}))
        self.training = TrainingConfig(**self._config.get("training", {}))
        self.augmentation = AugmentationConfig(**self._config.get("augmentation", {}))
        self.evaluation = EvaluationConfig(**self._config.get("evaluation", {}))


def load_config(config_path: str) -> Config:
    """
    Convenience function to load configuration from file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    return Config(config_path=config_path)

