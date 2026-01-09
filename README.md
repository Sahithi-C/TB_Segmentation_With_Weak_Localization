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
├── src/               # Source code
├── configs/           # Configuration files
├── notebooks/         # Jupyter notebooks
├── experiments/       # Experiment tracking
├── results/           # Model outputs and results
├── docs/              # Documentation
└── tests/             # Unit tests
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Quick Start

### 1. Configure Project

Edit `configs/base_config.yaml` with your settings.

### 2. Preprocess Data

```bash
python src/scripts/preprocess_data.py --config configs/base_config.yaml
```

### 3. Train Model

```bash
python src/scripts/train.py --config configs/training_config.yaml
```

### 4. Evaluate Model

```bash
python src/scripts/evaluate.py --model_path results/models/best_model.pth --config configs/eval_config.yaml
```

## Datasets

- **TBX11K**: Primary training dataset (~11,200 CXRs)
- **Shenzhen**: Cross-dataset validation
- **Montgomery**: Cross-dataset validation

## Key Features

- Weak localization mask generation
- Modality-specific pretraining
- Cross-dataset evaluation
- Class Activation Maps (CAMs) for interpretability
- Comprehensive metrics (Dice, IoU, statistical tests)

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

