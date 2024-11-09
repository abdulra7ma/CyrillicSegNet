import torch
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a results folder if it doesn't exist
os.makedirs('./results', exist_ok=True)

# ----------------------------------------
# Step 1: Define Transformations for Data
# ----------------------------------------

# Define transformations for the training and testing datasets
# Resize all images to 128x128 and normalize the pixel values
trans = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128 pixels
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize pixel values to [-1, 1]
])

# Assume dataset paths are defined: kyrgyz_train_path, kyrgyz_test_path
kyrgyz_train_path = './data/raw/handwritten_kyrgyz_letters/train'
kyrgyz_test_path = './data/raw/handwritten_kyrgyz_letters/test'

# The datasets and DataLoader are now using resized images.
train_data = torchvision.datasets.ImageFolder(root=kyrgyz_train_path, transform=trans)
test_data = torchvision.datasets.ImageFolder(root=kyrgyz_test_path, transform=trans)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# ----------------------------------------
# Step 2: Define Model Architecture
# ----------------------------------------

class Kyrgyz(nn.Module):
    def __init__(self):
        super(Kyrgyz, self).__init__()
        self.features = nn.Sequential(
            # conv1 : 3 * 128 * 128
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # conv2 : 16 * 63 * 63
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # conv3 : 32 * 30 * 30
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Flatten : 64 * 14 * 14
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*14*14, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=512, out_features=36)  # 36 classes for Kyrgyz letters
        )
    
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits
    
model = Kyrgyz()

# ----------------------------------------
# Step 3: Define Training and Evaluation Functions
# ----------------------------------------

def train(dataloader, model, loss_fn, optimizer):
    """Train the model for one epoch.
    
    Args:
        dataloader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer used for training.

    Returns:
        tuple: Average loss and top-1 accuracy for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    top1_acc = 0.0
    size = len(dataloader.dataset)

    # Iterate over batches of data
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        # Forward pass: compute model predictions
        pred = model(imgs)

        # Compute loss
        loss = loss_fn(pred, labels)

        # Backward pass: compute gradients and update parameters
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Accumulate loss and calculate top-1 accuracy
        total_loss += loss.item()
        predicted_1 = pred.argmax(1)
        top1_acc += (predicted_1 == labels).float().sum().item()

        # Log progress every 100 batches
        if batch_idx % 100 == 0:
            current = batch_idx * len(imgs)
            print(f"[{current:>5d}/{size:>5d}]")

    return total_loss / size, top1_acc / size

def test(dataloader, model, loss_fn):
    """Evaluate the model on validation or test data.
    
    Args:
        dataloader (DataLoader): DataLoader for validation or test data.
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function.

    Returns:
        tuple: Average loss and top-1 accuracy for the validation/test set.
    """
    model.eval()  # Set the model to evaluation mode
    size = len(dataloader.dataset)
    total_loss = 0.0
    top1_acc = 0.0

    # No gradient computation needed during evaluation
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass: compute model predictions
            pred = model(imgs)

            # Compute loss
            loss = loss_fn(pred, labels)

            # Accumulate loss and calculate top-1 accuracy
            total_loss += loss.item()
            predicted_1 = pred.argmax(1)
            top1_acc += (predicted_1 == labels).float().sum().item()

    return total_loss / size, top1_acc / size

# ----------------------------------------
# Step 4: Hyperparameters and Training Loop (with Best Practices)
# ----------------------------------------

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Learning rate scheduler: Reduce LR when a metric has stopped improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Early stopping configuration
patience = 5
best_val_loss = float('inf')
early_stopping_counter = 0

# Target accuracy for stopping training early if reached
target_accuracy = 0.99  # Stop training if validation accuracy reaches or exceeds 95%

# Track results
res = {
    "train_loss": [],
    "train_top1_acc": [],
    "val_loss": [],
    "val_top1_acc": [],
}

# Set the number of epochs
epochs = 20
best_model_path = './results/best_model.pth'

# Device configuration (use GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move model to the appropriate device
model.to(device)

# Training loop
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n{'-'*40}")
    
    # Train phase: Train the model using the training dataset
    train_loss, train_top1 = train(train_dataloader, model, loss_fn, optimizer)
    
    # Validation phase: Evaluate the model using the validation dataset
    val_loss, val_top1 = test(test_dataloader, model, loss_fn)
    
    # Save the results for the current epoch
    res['train_loss'].append(train_loss)
    res['train_top1_acc'].append(train_top1)
    res['val_loss'].append(val_loss)
    res['val_top1_acc'].append(val_top1)

    # Print epoch summary: Show the training and validation loss and accuracy
    print(f"Train Loss: {train_loss:.4f}, Train Top-1 Accuracy: {train_top1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Top-1 Accuracy: {val_top1:.4f}")
    
    # Learning rate scheduler step
    scheduler.step(val_loss)
    
    # Early stopping and model checkpointing
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model.")
        best_val_loss = val_loss
        early_stopping_counter = 0
        
        # Save the model as the best model so far
        torch.save(model.state_dict(), best_model_path)
    else:
        early_stopping_counter += 1
        print(f"No improvement in validation loss for {early_stopping_counter} consecutive epochs.")
    
    # Check if early stopping is needed based on patience
    if early_stopping_counter >= patience:
        print("Early stopping triggered due to no improvement. Stopping training.")
        break

    # Check if the target accuracy is reached to stop training
    if val_top1 >= target_accuracy:
        print(f"Target accuracy of {target_accuracy*100:.2f}% reached. Stopping training.")
        break

# Print final training status
print('Training Done!')

# Save the final model after all epochs or early stopping
torch.save(model.state_dict(), './results/final_model.pth')

# Save training results to a file
np.save('./results/training_results.npy', res)

# ----------------------------------------
# Step 5: Plot the Results
# ----------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(20, 7))
x = np.arange(len(res['train_loss']))

# Plot Training & Validation Loss
ax[0].plot(x, res['train_loss'], label='Training Loss')
ax[0].plot(x, res['val_loss'], label='Validation Loss')
ax[0].legend(loc='upper right')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training and Validation Loss')

# Plot Top-1 Accuracy
ax[1].plot(x, res['train_top1_acc'], label='Training Top-1 Accuracy')
ax[1].plot(x, res['val_top1_acc'], label='Validation Top-1 Accuracy')
ax[1].legend(loc='upper left')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Top-1 Accuracy')
ax[1].set_title('Training and Validation Top-1 Accuracy')

# Save the plot to the results folder
plt.savefig('./results/training_plot.png')
plt.show()
