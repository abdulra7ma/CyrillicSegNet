# Model Training Guide

This guide provides detailed information about training the Kyrgyz Letter Recognition model with various configurations and parameters.

## Quick Start

Basic training with default parameters:
```bash
python src/train_model.py --epochs 100 --batch-size 32
```

## Training Parameters

### Basic Parameters

```bash
python src/train_model.py \
    --epochs 100 \              # Number of training epochs
    --batch-size 32 \          # Batch size for training
    --learning-rate 0.001 \    # Initial learning rate
    --weight-decay 1e-4        # L2 regularization factor
```

### Model Architecture Parameters

```bash
python src/train_model.py \
    --dropout 0.5 \            # Dropout rate for regularization
    --hidden-size 2048 \       # Size of hidden layers
    --image-size 128           # Input image dimensions
```

### Training Control Parameters

```bash
python src/train_model.py \
    --target-accuracy 0.99 \   # Target accuracy to stop training
    --patience 5 \             # Early stopping patience
    --scheduler-patience 3 \    # Learning rate scheduler patience
    --scheduler-factor 0.5     # Learning rate reduction factor
```

### Data and Output Parameters

```bash
python src/train_model.py \
    --train-path ./data/raw/handwritten_kyrgyz_letters/train \
    --test-path ./data/raw/handwritten_kyrgyz_letters/test \
    --output-dir ./results \
    --model-name best_model.pth
```

## Parameter Details

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Total number of training epochs |
| `batch-size` | 32 | Number of samples per training batch |
| `learning-rate` | 0.001 | Initial learning rate for Adam optimizer |
| `weight-decay` | 1e-4 | L2 regularization strength |

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dropout` | 0.5 | Dropout probability for regularization |
| `hidden-size` | 2048 | Number of neurons in hidden layers |
| `image-size` | 128 | Input image dimensions (width=height) |

### Training Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target-accuracy` | 0.99 | Training stops when reaching this accuracy |
| `patience` | 5 | Number of epochs to wait before early stopping |
| `scheduler-patience` | 3 | Epochs to wait before reducing learning rate |
| `scheduler-factor` | 0.5 | Factor to reduce learning rate by |

### Paths and Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train-path` | ./data/raw/handwritten_kyrgyz_letters/train | Training data directory |
| `test-path` | ./data/raw/handwritten_kyrgyz_letters/test | Testing data directory |
| `output-dir` | ./results | Directory for saving outputs |
| `model-name` | best_model.pth | Name of saved model file |

## Training Configurations

### Basic Training

For basic model training with default parameters:
```bash
python src/train_model.py --epochs 100 --batch-size 32
```

### High-Performance Configuration

For maximum accuracy (slower training):
```bash
python src/train_model.py \
    --epochs 200 \
    --batch-size 16 \
    --learning-rate 0.0005 \
    --weight-decay 1e-5 \
    --dropout 0.6 \
    --hidden-size 4096 \
    --scheduler-patience 5 \
    --target-accuracy 0.995
```

### Fast Training Configuration

For quick training and prototyping:
```bash
python src/train_model.py \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.002 \
    --dropout 0.3 \
    --hidden-size 1024 \
    --scheduler-patience 2 \
    --target-accuracy 0.95
```

### Memory-Efficient Configuration

For training with limited memory:
```bash
python src/train_model.py \
    --epochs 150 \
    --batch-size 8 \
    --hidden-size 1024 \
    --image-size 96
```

## Training Output

The training script generates several outputs in the specified output directory:

1. **Model Checkpoints**
   - `best_model.pth`: Model with best validation performance
   - `final_model.pth`: Model state after training completion

2. **Training Statistics**
   - `training_results.npy`: NumPy file containing training metrics
   - `training_plot.png`: Plot showing loss and accuracy curves

3. **Console Output**
   - Training progress per epoch
   - Validation metrics
   - Early stopping information
   - Learning rate adjustments

## Monitoring Training

During training, the script outputs:
```
Epoch 1/100
----------------------------------------
[    0/50000]
Train Loss: 2.3456, Train Top-1 Accuracy: 0.4567
Val Loss: 2.1234, Val Top-1 Accuracy: 0.4789
```

## Tips for Best Results

1. **Finding Optimal Learning Rate**:
   ```bash
   # Start with a larger learning rate and reduce if training is unstable
   python src/train_model.py --learning-rate 0.01
   # If unstable, try
   python src/train_model.py --learning-rate 0.001
   ```

2. **Balancing Batch Size and Memory**:
   ```bash
   # For 12GB GPU memory
   python src/train_model.py --batch-size 32 --hidden-size 2048
   # For 8GB GPU memory
   python src/train_model.py --batch-size 16 --hidden-size 1024
   ```

3. **Preventing Overfitting**:
   ```bash
   # Increase regularization
   python src/train_model.py --dropout 0.6 --weight-decay 1e-4
   ```

4. **Improving Accuracy**:
   ```bash
   # Longer training with gradual learning rate reduction
   python src/train_model.py \
       --epochs 300 \
       --scheduler-patience 5 \
       --scheduler-factor 0.7
   ```

## Troubleshooting

1. **Out of Memory Errors**
   - Reduce `batch-size`
   - Reduce `hidden-size`
   - Reduce `image-size`

2. **Slow Training**
   - Increase `batch-size`
   - Reduce `hidden-size`
   - Remove `weight-decay`

3. **Poor Convergence**
   - Adjust `learning-rate`
   - Increase `patience`
   - Modify `scheduler-factor`

4. **Overfitting**
   - Increase `dropout`
   - Increase `weight-decay`
   - Reduce `hidden-size`