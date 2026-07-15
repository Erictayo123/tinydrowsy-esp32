"""
train.py
--------
Train TinyEyeNetV2 on the processed dataset.

Usage:
    python src/training/train.py --config src/training/config.yaml

Outputs:
    - training_output/best_model.pth
    - training_output/training_curves.png
    - training_output/confusion_matrix.png
    - training_output/classification_report.txt
    - training_output/history.json
"""

import os
import sys
import json
from pathlib import Path
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import TinyDrowsyDataset
from data.augmentations import get_train_transform, get_validation_transform
from models.model import TinyEyeNetV2, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyDrowsy model")
    parser.add_argument("--config", type=str, default="src/training/config.yaml",
                        help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Paths
    data_root = Path(config['dataset']['root'])
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Dataset
    print("📊 Loading datasets...")
    train_dataset = TinyDrowsyDataset(
        data_root / "train",
        transform=get_train_transform()
    )
    val_dataset = TinyDrowsyDataset(
        data_root / "validation",
        transform=get_validation_transform()
    )
    test_dataset = TinyDrowsyDataset(
        data_root / "test",
        transform=get_validation_transform()
    )
    
    class_names = train_dataset.class_names()
    num_classes = train_dataset.num_classes()
    
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # DataLoaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    print("🧠 Creating model...")
    model = TinyEyeNetV2(
        num_classes=num_classes,
        input_channels=config['dataset']['channels']
    ).to(device)
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:2d}/{config['training']['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_names': class_names,
            }, output_dir / 'best_model.pth')
            print(f"  ✅ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break
    
    # Save training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    print("📊 Training curves saved")
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\n" + report)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format='d')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Outputs saved to: {output_dir}")
    print(f"  - best_model.pth")
    print(f"  - training_curves.png")
    print(f"  - confusion_matrix.png")
    print(f"  - classification_report.txt")
    print(f"  - history.json")


if __name__ == "__main__":
    main()