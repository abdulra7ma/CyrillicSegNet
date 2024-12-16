# Kyrgyz Letter Recognition Project

This project implements a machine learning model for recognizing Kyrgyz letters and words, including both handwritten and typed Cyrillic text. It includes both training components and a user interface for real-time recognition.

## Project Structure

```
├── README.md                 # Project documentation
├── app_ui/                   # User interface components
├── data/                     # Training and test datasets
├── notebooks/               # Development and analysis notebooks
├── results/                 # Trained models and evaluation results
└── src/                    # Source code for training and inference
```

## Data

The training data consists of three main components:
1. Handwritten Kyrgyz letters dataset
   - Location: `data/raw/handwritten_kyrgyz_letters/`
   - Format: Individual PNG images (28x28 pixels)
   - Size: ~10,000 samples across 36 classes

2. Cyrillic typed words dataset
   - Location: `data/raw/cyrilic_words/`
   - Format: Text files with corresponding images
   - Size: ~5,000 word samples

3. Combined dataset (processed)
   - Location: `data/processed/combined/`
   - Contains aligned and preprocessed versions of both datasets

## Setup

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for training)

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements File
Create a `requirements.txt` file with these dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.4.0
opencv-python>=4.7.0
pillow>=9.0.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
PyQt5>=5.15.0
```

## Training

The project supports multiple training configurations for different use cases. For detailed training documentation and parameter configurations, see [ModelTraining.md](docs/ModelTraining.md).

### Quick Start


1. Basic training with default configuration:
```bash
python src/train_model.py --epochs 100 --batch-size 32
```

### Common Training Configurations

1. High-Performance Training:
```bash
python src/train_model.py \
    --epochs 200 \
    --batch-size 16 \
    --learning-rate 0.0005 \
    --hidden-size 4096
```

2. Fast Prototype Training:
```bash
python src/train_model.py \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.002
```

3. Memory-Efficient Training:
```bash
python src/train_model.py \
    --epochs 150 \
    --batch-size 8 \
    --hidden-size 1024
```

### Training Output

The training process generates:
- Model checkpoints (`best_model.pth`, `final_model.pth`)
- Training statistics and plots
- Validation metrics

All outputs are saved in the `results/` directory.

### Key Parameters

- `epochs`: Number of training cycles
- `batch-size`: Samples per training batch
- `learning-rate`: Controls optimization speed
- `hidden-size`: Model capacity
- `dropout`: Regularization strength

For a complete list of parameters and advanced configurations, refer to [ModelTraining.md](docs/ModelTraining.md).


## User Interface

To launch the recognition interface:

```bash
python app_ui/kyrgyz_recognition_gui.py
```

The UI provides:
- Image upload functionality
- Real-time recognition
- Confidence scores for predictions
- Support for both single letter and word recognition

## Model Architecture

The project uses a CNN-based architecture with:
- 3 convolutional layers
- 2 fully connected layers
- Dropout for regularization
- BatchNormalization
- ReLU activation

Full model specifications are available in `src/model.py`.

## Results

Training results and model checkpoints are saved in:
- `results/models/`: Trained model weights
- `results/graphs/`: Training progress visualization
- `results/training_results.npy`: Detailed metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kyrgyz Language Institute for providing baseline datasets
- Contributors to the handwritten letters dataset
- OpenCV and PyTorch communities

## Contact

For questions and support, please open an issue in the GitHub repository.

---

**Note**: This project is part of ongoing research in Kyrgyz language OCR systems. Please cite this work if you use it in your research.