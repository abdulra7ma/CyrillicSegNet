import torch
import os
import numpy as np
import argparse
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for Kyrgyz Letter Recognition')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--hidden-size', type=int, default=2048, help='Size of hidden layer')
    
    # Data parameters
    parser.add_argument('--image-size', type=int, default=128, help='Input image size')
    parser.add_argument('--train-path', type=str, default='./data/raw/handwritten_kyrgyz_letters/train',
                        help='Path to training data')
    parser.add_argument('--test-path', type=str, default='./data/raw/handwritten_kyrgyz_letters/test',
                        help='Path to test data')
    
    # Training control
    parser.add_argument('--target-accuracy', type=float, default=0.99, help='Target accuracy to stop training')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--scheduler-patience', type=int, default=3, help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help='Factor for learning rate scheduler')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory for saving results')
    parser.add_argument('--model-name', type=str, default='best_model.pth', help='Name for saved model')
    
    return parser.parse_args()

class Kyrgyz(nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_size=2048):
        super(Kyrgyz, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(64*14*14, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_size, hidden_size//4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_size//4, 36)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    top1_acc = 0.0
    size = len(dataloader.dataset)
    
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        pred = model(imgs)
        loss = loss_fn(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted_1 = pred.argmax(1)
        top1_acc += (predicted_1 == labels).float().sum().item()
        
        if batch_idx % 100 == 0:
            current = batch_idx * len(imgs)
            print(f"[{current:>5d}/{size:>5d}]")
            
    return total_loss / size, top1_acc / size

def test(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    top1_acc = 0.0
    size = len(dataloader.dataset)
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, labels)
            
            total_loss += loss.item()
            predicted_1 = pred.argmax(1)
            top1_acc += (predicted_1 == labels).float().sum().item()
            
    return total_loss / size, top1_acc / size

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define transformations
    trans = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Load datasets
    train_data = torchvision.datasets.ImageFolder(root=args.train_path, transform=trans)
    test_data = torchvision.datasets.ImageFolder(root=args.test_path, transform=trans)
    
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = Kyrgyz(dropout_rate=args.dropout, hidden_size=args.hidden_size).to(device)
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        verbose=True
    )
    
    # Training tracking
    best_val_loss = float('inf')
    early_stopping_counter = 0
    res = {
        "train_loss": [], "train_top1_acc": [],
        "val_loss": [], "val_top1_acc": []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}\n{'-'*40}")
        
        train_loss, train_top1 = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_top1 = test(test_dataloader, model, loss_fn, device)
        
        # Save results
        res['train_loss'].append(train_loss)
        res['train_top1_acc'].append(train_top1)
        res['val_loss'].append(val_loss)
        res['val_top1_acc'].append(val_top1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Top-1 Accuracy: {train_top1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Top-1 Accuracy: {val_top1:.4f}")
        
        scheduler.step(val_loss)
        
        # Model checkpointing
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.model_name))
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epochs")
        
        # Early stopping checks
        if early_stopping_counter >= args.patience:
            print("Early stopping triggered")
            break
        
        if val_top1 >= args.target_accuracy:
            print(f"Reached target accuracy of {args.target_accuracy*100:.2f}%")
            break
    
    # Save final results
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_' + args.model_name))
    np.save(os.path.join(args.output_dir, 'training_results.npy'), res)
    
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    x = np.arange(len(res['train_loss']))
    
    ax[0].plot(x, res['train_loss'], label='Training Loss')
    ax[0].plot(x, res['val_loss'], label='Validation Loss')
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    
    ax[1].plot(x, res['train_top1_acc'], label='Training Top-1 Accuracy')
    ax[1].plot(x, res['val_top1_acc'], label='Validation Top-1 Accuracy')
    ax[1].legend(loc='upper left')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Top-1 Accuracy')
    ax[1].set_title('Training and Validation Top-1 Accuracy')
    
    plt.savefig(os.path.join(args.output_dir, 'training_plot.png'))
    plt.close()

if __name__ == "__main__":
    main()