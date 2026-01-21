# TB Segmentation: Generalizing Tuberculosis Segmentation in Chest X-rays

A deep learning project for segmenting tuberculosis-consistent regions in chest X-rays using modality-specific U-Nets and weak localization techniques.

## Project Overview

This project aims to develop a generalizable deep learning model capable of accurately segmenting TB-consistent regions in chest X-rays (CXRs). The approach combines:

- **Modality-specific U-Net architectures** - Fine-tuned on CXR data
- **Weak localization techniques** - Training with image-level labels instead of pixel-level annotations
- **Cross-dataset evaluation** - Ensuring robustness across diverse imaging sources

## Project Structure

```
tb-segmentation/
├── data/              # Dataset storage
│   ├── raw/          # Raw datasets (TBX11K, Shenzhen, Montgomery)
│   ├── processed/    # Processed data
│   └── weak_masks/   # Generated weak localization masks
├── src/               # Source code
│   ├── data/         # Data processing modules
│   │   └── weak_localization.py  # Bbox-to-mask conversion
│   ├── models/       # Model architectures
│   ├── training/     # Training utilities
│   ├── evaluation/  # Evaluation metrics
│   └── utils/       # Helper utilities
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks
│   ├── 01_download_datasets_to_drive.ipynb
│   ├── 02_explore_datasets.ipynb
│   └── 03_generate_weak_masks.ipynb
├── experiments/       # Experiment tracking
├── results/           # Model outputs and results
├── docs/              # Documentation
└── tests/             # Unit tests
```

## Datasets

- **TBX11K**: Primary training dataset (~11,200 CXRs)
- **Shenzhen**: Cross-dataset validation
- **Montgomery**: Cross-dataset validation

## Key Features

- **Weak localization mask generation** - Convert bounding box annotations to segmentation masks
- Modality-specific pretraining
- Cross-dataset evaluation
- Class Activation Maps (CAMs) for interpretability
- Comprehensive metrics (Dice, IoU, statistical tests)

## Current Status

### ✅ Completed
- **Weak Localization Module** (`src/data/weak_localization.py`)
  - Bounding box to mask conversion
  - Batch processing for entire datasets
  - Visualization utilities
  - Successfully generated 8,399 masks from TBX11K dataset

### 🚧 In Progress
- Data preprocessing pipeline
- PyTorch Dataset class implementation

### 📋 Planned
- U-Net model architecture
- Training framework
- Cross-dataset evaluation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)
- 16GB+ RAM

## Project Timeline

- **Phase 1**: Foundation Setup (Week 1)
- **Phase 2**: Data Pipeline (Week 2)
- **Phase 3**: Model Architecture (Weeks 3-4)
- **Phase 4**: Training Framework (Weeks 5-6)
- **Phase 5**: Evaluation Framework (Weeks 7-8)
- **Phase 6**: Advanced Features (Weeks 9-10)
- **Phase 7**: Integration & Testing (Weeks 11-12)

## Author

Sahithi C

## License

This project is for academic research purposes.

## References

See `docs/methodology.md` for detailed references and methodology.

