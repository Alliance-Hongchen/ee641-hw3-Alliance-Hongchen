"""
Training script for sorting detection task.

Trains all three positional encoding variants and compares them.

Mostly PROVIDED to students (they may need to tune hyperparameters).
"""

import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.dataset import create_dataloaders
from src.model import create_model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: TransformerEncoder model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Classification accuracy
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for sequences, labels, lengths in pbar:
        sequences, labels = sequences.to(device), labels.to(device)

        # Forward pass
        logits = model(sequences)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += len(labels)

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct / len(labels):.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.

    Args:
        model: TransformerEncoder model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        avg_loss: Average loss
        accuracy: Classification accuracy
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for sequences, labels, lengths in pbar:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            logits = model(sequences)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def plot_training_curves(history, save_path):
    """
    Plot and save training curves for a single model.

    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def train_model(encoding_type, config, train_loader, val_loader, device):
    """
    Train a single model variant.

    Args:
        encoding_type: One of 'sinusoidal', 'learned', 'none'
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on

    Returns:
        model: Trained model
        history: Training history dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Training with {encoding_type.upper()} positional encoding")
    print(f"{'=' * 60}")

    # Create model
    model = create_model(
        encoding_type=encoding_type,
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch:2d}/{config['num_epochs']} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model
            model_dir = os.path.join(config['results_dir'], encoding_type)
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
                'encoding_type': encoding_type
            }, os.path.join(model_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping after {epoch} epochs")
                break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")

    # Load best model
    checkpoint = torch.load(os.path.join(config['results_dir'], encoding_type, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save training curves
    model_dir = os.path.join(config['results_dir'], encoding_type)
    plot_training_curves(history, os.path.join(model_dir, 'training_curves.png'))

    # Save history
    history['total_time'] = total_time
    history['encoding_type'] = encoding_type
    with open(os.path.join(model_dir, 'training_log.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train sorting detection models')
    parser.add_argument('--encoding', type=str, default='all',
                        choices=['sinusoidal', 'learned', 'none', 'all'],
                        help='Positional encoding type to train (default: all)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Configuration
    config = {
        'data_dir': '../data/problem2',
        'results_dir': 'results',
        'vocab_size': 100,  # Integers 0-99
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 4,
        'd_ff': 256,
        'dropout': 0.1,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': 0.0005,
        'patience': 5,
        'max_seq_len': 512,  # For learned encoding
    }

    print("\n" + "=" * 60)
    print("Training Sorting Detection Transformer")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)

    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size']
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Determine which models to train
    if args.encoding == 'all':
        encoding_types = ['sinusoidal', 'learned', 'none']
    else:
        encoding_types = [args.encoding]

    # Train models
    results = {}
    for encoding_type in encoding_types:
        model, history = train_model(encoding_type, config, train_loader, val_loader, device)
        results[encoding_type] = {
            'model': model,
            'history': history
        }

    # Summary comparison
    if len(encoding_types) > 1:
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"{'Encoding':15s} | {'Val Acc':8s} | {'Val Loss':8s} | {'Time (s)':8s}")
        print("-" * 60)
        for enc_type in encoding_types:
            hist = results[enc_type]['history']
            best_acc = max(hist['val_acc'])
            best_loss = min(hist['val_loss'])
            total_time = hist['total_time']
            print(f"{enc_type:15s} | {best_acc:8.4f} | {best_loss:8.4f} | {total_time:8.1f}")
        print("=" * 60)

    print(f"\nResults saved to {config['results_dir']}/")
    print("  - sinusoidal/")
    print("  - learned/")
    print("  - none/")


if __name__ == '__main__':
    main()
