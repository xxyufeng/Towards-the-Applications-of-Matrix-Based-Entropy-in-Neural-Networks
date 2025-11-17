import torch
import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision.models import vit_b_16  # Import a pre-trained ViT model
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm
from dataset import load_data_libsvm
from model import VisionTransformer
import csv

# Set random seeds for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = 42 
seed_everything(seed)

# Load MNIST dataset and preprocess
def load_data(split, dataset_name, datadir, nclasses):
    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    if dataset_name == 'SVHN':
        get_dataset = getattr(datasets, dataset_name)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    elif dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST']:
        get_dataset = getattr(datasets, dataset_name)
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
    else:
        dataset = load_data_libsvm(dataset_name, datadir, split)

    return dataset

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    iteration = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for imgs, labels in pbar:
        # Move data to device (GPU/CPU)
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(imgs)[0]
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += imgs.size(0)

        # Compute Mutual Information
        model.eval()
        current_iteration = epoch * len(train_loader) + iteration
        with torch.no_grad():
            model.compute_mi(imgs, labels, model, current_iteration)
        
        # Update progress bar
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        I_T_X = model.MI[current_iteration, -1, 0]
        I_T_Y = model.MI[current_iteration, -1, 1]
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.4f}', 'I(T;X)': f'{I_T_X:.4f}', 'I(T;Y)': f'{I_T_Y:.4f}'})

        iteration += 1
        model.train()
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='[Validation]')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(imgs)[0]
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)
            
            # Collect preds and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.4f}'})
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20):
    """Full training pipeline"""
    # Record training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"Training started (Device: {device})")
    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        #Show I(T;X) and I(T;Y) for the last iteration of this epoch
        I_T_X = model.MI[(epoch+1) * len(train_loader) - 1, -1, 0]
        I_T_Y = model.MI[(epoch+1) * len(train_loader) - 1, -1, 1]
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | I(T;X): {I_T_X:.4f} | I(T;Y): {I_T_Y:.4f}')
    
    # Save trained model
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print(f"Model saved as 'vit_cifar10.pth'")
    
    return history, all_preds, all_labels


if __name__ == "__main__":
    # Configuration (adjust based on server resources)
    BATCH_SIZE = 256
    IMG_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5  # Regularization to prevent overfitting
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET = 'CIFAR10'
    DATADIR = './datasets'
    OUTPUT = 'vit_cifar10_mi_results_1116.csv'
    HISTORY = 'vit_cifar10_training_history_1116.csv'
    NCHANNELS = 3  # RGB images
    
    # Step 1: Load data
    train_dataset = load_data('train', DATASET, DATADIR, NCHANNELS)
    val_dataset = load_data('val', DATASET, DATADIR, NCHANNELS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # Step 2: Initialize ViT model
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=4,
        in_ch=3,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_dim=1024,
        num_classes=10,
        dropout=0.1,
        n_iterations=EPOCHS * (len(train_loader))
    )
    model.to(DEVICE)
    print(f"ViT model initialized. Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Classification task
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # Learning rate scheduler (reduce LR when val loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Step 4: Train model
    history, all_preds, all_labels = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, EPOCHS
    )

    # Step 5: Report MI results
    file_exists = os.path.isfile(OUTPUT)
    # Save MI tensor to CSV, header order:
    # ['Iteration', 'layer_FMSA: I(X;T)', 'layer_FMSA: I(T;Y)',
    #  'layer_CLS: I(X;T)', 'layer_CLS: I(T;Y)',
    #  'layer_LAST: I(X;T)', 'layer_LAST: I(T;Y)']
    with open(OUTPUT, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Iteration',
                             'layer_FMSA_mid: I(X;T)', 'layer_FMSA_mid: I(T;Y)',
                             'layer_FMSA: I(X;T)', 'layer_FMSA: I(T;Y)',
                             'layer_CLS: I(X;T)', 'layer_CLS: I(T;Y)',
                             'layer_LAST: I(X;T)', 'layer_LAST: I(T;Y)'])

        # model.MI is expected shape (n_iterations, 3, 2)
        if isinstance(model.MI, torch.Tensor):
            mi_arr = model.MI.detach().cpu().numpy()
        else:
            mi_arr = np.array(model.MI)

        n_iter = mi_arr.shape[0]
        for it in range(n_iter):
            # flatten in the order: layer0 I(X;T), layer0 I(T;Y), layer1 I(X;T), ...
            row = [it] + [float(mi_arr[it, layer, col]) for layer in range(4) for col in range(2)]
            writer.writerow(row)

    with open(HISTORY, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
        for epoch in range(EPOCHS):
            writer.writerow([
                epoch + 1,
                history['train_loss'][epoch],
                history['train_acc'][epoch],
                history['val_loss'][epoch],
                history['val_acc'][epoch]
            ])

    print(f'Results and history saved to {OUTPUT} and {HISTORY} respectively.')