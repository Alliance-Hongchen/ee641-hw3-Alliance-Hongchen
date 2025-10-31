"""
Training script for sequence-to-sequence transformer.

PARTIALLY PROVIDED - You need to implement key training components.

This file provides the overall training structure, but you must implement:
- Loss computation with padding mask
- Accuracy calculation
- Some training loop details
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import create_dataloaders
from src.model import Seq2SeqTransformer
from src.attention import create_causal_mask


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - Loss Computation with Masking
# ============================================================================

def compute_masked_loss(logits, targets, pad_idx=0):
    """
    Compute cross-entropy loss while ignoring padding tokens.

    When we pad sequences to the same length, we don't want the model to
    learn from padding positions. This function masks out the padding tokens
    when computing loss.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]
        pad_idx: Padding token ID (default: 0)

    Returns:
        loss: Scalar loss value (averaged over non-padding tokens)

    Implementation steps:
        1. Flatten logits and targets for cross_entropy
        2. Compute loss with reduction='none' (per-token losses)
        3. Create mask for non-padding positions
        4. Apply mask and take mean of non-padding losses

    Hints:
        - torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        - Create mask: (targets != pad_idx).float()
        - Use mask to select non-padding losses
        - Remember to reshape logits to [batch * seq_len, vocab_size]
    """
    # TODO: Reshape logits for cross entropy
    logits_flat = logits.reshape(-1, logits.size(-1))  # [batch * seq_len, vocab_size]
    targets_flat = targets.reshape(-1)  # [batch * seq_len]

    # TODO: Compute per-token loss (reduction='none')
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    # TODO: Create mask for non-padding tokens
    mask = (targets_flat != pad_idx).float()

    # TODO: Apply mask and compute mean
    masked_loss = (loss * mask).sum() / mask.sum()

    return masked_loss
    # raise NotImplementedError("Implement compute_masked_loss")


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - Accuracy Calculation
# ============================================================================

def calculate_accuracy(logits, targets, pad_idx=0):
    """
    Calculate token-level and sequence-level accuracy, ignoring padding.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]
        pad_idx: Padding token ID

    Returns:
        token_acc: Fraction of correctly predicted non-padding tokens
        seq_acc: Fraction of sequences where ALL tokens are correct

    Implementation steps:
        1. Get predictions from logits (argmax)
        2. Create mask for non-padding positions
        3. Calculate token accuracy (correct / total non-padding)
        4. Calculate sequence accuracy (all tokens correct in sequence)
    """
    # TODO: Get predictions
    predictions = logits.argmax(dim=-1)  # [batch, seq_len]

    # TODO: Create mask for non-padding tokens
    mask = (targets != pad_idx)

    # TODO: Calculate token accuracy
    correct_tokens = ((predictions == targets) & mask).sum().item()
    total_tokens = mask.sum().item()
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    # TODO: Calculate sequence accuracy
    # All tokens in sequence must be correct (considering only non-padding)
    # For each sequence, check if all non-padding tokens match
    seq_correct = ((predictions == targets) | ~mask).all(dim=-1)
    seq_acc = seq_correct.float().mean().item()

    return token_acc, seq_acc
    # raise NotImplementedError("Implement calculate_accuracy")


# ============================================================================
# Training Functions (PROVIDED with TODOs)
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, pad_idx=0):
    """
    Train for one epoch.

    MOSTLY PROVIDED - Uses your loss and accuracy functions.
    """
    model.train()
    total_loss = 0
    total_token_acc = 0
    total_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)
        batch_size, seq_len = tgt.shape

        # Create causal mask for decoder
        tgt_mask = create_causal_mask(seq_len, device=device)

        # Prepare decoder input and target
        # Decoder input: shift right (don't include last token)
        # Target: shift left (don't include first token / start token)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Adjust mask for shifted sequence
        tgt_mask_shifted = create_causal_mask(seq_len - 1, device=device)

        # Forward pass
        logits = model(src, tgt_input, tgt_mask=tgt_mask_shifted)

        # Compute loss (uses your implementation!)
        try:
            loss = compute_masked_loss(logits, tgt_output, pad_idx=pad_idx)
        except NotImplementedError:
            # Fallback to simple loss if not implemented yet
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = nn.functional.cross_entropy(logits_flat, tgt_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate accuracy (uses your implementation!)
        try:
            token_acc, _ = calculate_accuracy(logits, tgt_output, pad_idx=pad_idx)
        except NotImplementedError:
            predictions = logits.argmax(dim=-1)
            token_acc = (predictions == tgt_output).float().mean().item()

        total_loss += loss.item()
        total_token_acc += token_acc
        total_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{token_acc:.4f}'
        })

    avg_loss = total_loss / total_batches
    avg_acc = total_token_acc / total_batches

    return avg_loss, avg_acc


def evaluate(model, dataloader, device, pad_idx=0):
    """
    Evaluate model on validation/test set.

    PROVIDED COMPLETE.
    """
    model.eval()
    total_loss = 0
    total_token_acc = 0
    total_seq_acc = 0
    total_batches = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            seq_len = tgt.size(1)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = create_causal_mask(seq_len - 1, device=device)

            logits = model(src, tgt_input, tgt_mask=tgt_mask)

            try:
                loss = compute_masked_loss(logits, tgt_output, pad_idx=pad_idx)
                token_acc, seq_acc = calculate_accuracy(logits, tgt_output, pad_idx=pad_idx)
            except NotImplementedError:
                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_flat = tgt_output.reshape(-1)
                loss = nn.functional.cross_entropy(logits_flat, tgt_flat)
                predictions = logits.argmax(dim=-1)
                token_acc = (predictions == tgt_output).float().mean().item()
                seq_acc = (predictions == tgt_output).all(dim=1).float().mean().item()

            total_loss += loss.item()
            total_token_acc += token_acc
            total_seq_acc += seq_acc
            total_batches += 1

    avg_loss = total_loss / total_batches
    avg_token_acc = total_token_acc / total_batches
    avg_seq_acc = total_seq_acc / total_batches

    return avg_loss, avg_token_acc, avg_seq_acc


def plot_training_curves(history, save_path):
    """Plot training curves. PROVIDED COMPLETE."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Token accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Token Accuracy')
    axes[1].set_title('Token Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Sequence accuracy
    axes[2].plot(epochs, history['val_seq_acc'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Sequence Accuracy')
    axes[2].set_title('Sequence Accuracy')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Loop (PROVIDED)
# ============================================================================

def main():
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
        'data_dir': '../data/problem1',
        'results_dir': 'results',
        'vocab_size': 12,  # 0-9 digits + '+' token + pad token
        'd_model': 128,
        'num_heads': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'd_ff': 512,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 0.001,
    }

    print("=" * 60)
    print("Training Multi-Digit Addition Transformer")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)

    # Create results directory
    os.makedirs(config['results_dir'], exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size']
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = Seq2SeqTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    # verbose=True may be not used in new pytorch
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_seq_acc': []
    }

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, val_acc, val_seq_acc = evaluate(model, val_loader, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_seq_acc'].append(val_seq_acc)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:3d}/{config['num_epochs']} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Seq Acc: {val_seq_acc:.4f}")

        # Plot training curves
        plot_training_curves(history, os.path.join(config['results_dir'], 'training_curves.png'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(config['results_dir'], 'best_model.pth'))
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f}m)")

    # Test
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(config['results_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_seq_acc = evaluate(model, test_loader, device)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Token Accuracy: {test_acc:.4f}")
    print(f"  Test Sequence Accuracy: {test_seq_acc:.4f}")

    # Save training history
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['test_seq_acc'] = test_seq_acc
    history['config'] = config

    with open(os.path.join(config['results_dir'], 'training_log.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to {config['results_dir']}/")


if __name__ == '__main__':
    main()
