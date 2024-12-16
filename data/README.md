# Kyrgyz Letter Recognition Project

A deep learning project for recognizing Kyrgyz letters and words using computer vision and neural networks. This system supports both handwritten and typed Cyrillic text recognition with a user-friendly GUI interface.

## Overview

This project implements an OCR (Optical Character Recognition) system specifically designed for the Kyrgyz language, supporting both individual letter recognition and full word processing. It combines traditional computer vision techniques with modern deep learning approaches to achieve high accuracy in Kyrgyz text recognition.

## Features

- Individual letter recognition for 36 Kyrgyz letters
- Full word recognition capability
- Real-time recognition through GUI
- Support for both handwritten and typed text
- Preprocessing pipeline for image optimization
- Extensive visualization tools for analysis

## Project Structure

```
├── README.md                 # Main project documentation
├── app_ui/                   # User interface components
│   ├── kyrgyz_recognition_gui.py
│   └── word_recogination.py
├── data/                     # Training and test datasets
│   ├── README.md            # Detailed data documentation
│   ├── raw/                 # Original datasets
│   ├── processed/           # Processed data
│   └── test/                # Test datasets
├── notebooks/               # Analysis and development notebooks
├── results/                 # Trained models and evaluations
└── src/                     # Source code
    ├── model.py
    ├── train_model.py
    └── letter_recog.py
```

## Datasets

This project utilizes two main datasets:

1. **Handwritten Kyrgyz Letters Dataset**
   - 36 classes of Kyrgyz letters
   - Over 10,000 samples
   - [Dataset on Kaggle](https://www.kaggle.com/datasets/ilgizzhumaev/database-of-36-handwritten-kyrgyz-letters/code)

2. **Cyrillic Handwriting Dataset**
   - Comprehensive Cyrillic word samples
   - [Dataset on Kaggle](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset/data)

For detailed information about data organization and setup, please refer to [data/README.md](data/README.md).

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- 8GB RAM minimum (16GB recommended)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/abdulra7ma/CyrillicSegNet.git
cd CyrillicSegNet
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and setup datasets (follow instructions in data/README.md)

5. Run preprocessing:
```bash
python src/data_preprocessing.py
```

## Model Architecture

The system uses a CNN-based architecture with the following key components:

- Input Layer: 28x28 grayscale images
- Convolutional Layers: 3 layers with increasing filters
- Batch Normalization after each conv layer
- Max Pooling layers for dimensionality reduction
- Dropout layers (0.5) for regularization
- Dense layers with ReLU activation
- Output Layer: Softmax with 36 classes

## Training

1. Prepare the datasets according to data/README.md
2. Configure training parameters in src/config.py
3. Start training:
```bash
python src/train_model.py --epochs 100 --batch-size 32
```

Training metrics and checkpoints are saved in the results/ directory.

## Usage

### GUI Application
```bash
python app_ui/kyrgyz_recognition_gui.py
```

The GUI provides:
- Image upload functionality
- Real-time recognition
- Confidence scores
- Support for both single letter and word recognition

### Command Line Interface
```bash
python src/letter_recog.py --image path/to/image.png
```

## Results and Performance

- Letter Recognition Accuracy: 94.5%
- Word Recognition Accuracy: 89.2%
- Average Processing Time: 0.3s per image

Detailed performance metrics are available in the notebooks/ directory.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
We follow PEP 8 guidelines. Format code using:
```bash
black .
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{kyrgyz_letter_recognition,
  title = {Kyrgyz Letter Recognition Project},
  year = {2024},
  author = {Your Name},
  url = {https://github.com/yourusername/kyrgyz-letter-recognition}
}
```

## Acknowledgments

- Kyrgyz Language Institute for dataset support
- Contributors to the handwritten letters dataset
- PyTorch and OpenCV communities

## Contact

For questions and support:
- Open an issue in the GitHub repository
- Email: your.email@example.com

## Documentation

Full documentation is available in the /docs directory and includes:
- API Reference
- Model Architecture Details
- Training Guidelines
- Best Practices

For dataset-specific documentation, see [data/README.md](data/README.md).