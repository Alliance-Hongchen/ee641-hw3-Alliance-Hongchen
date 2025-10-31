"""
Dataset loader for sorting detection task.

This file will be PROVIDED to students (minimal modifications needed).
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader


class SortingDetectionDataset(Dataset):
    """
    Dataset for sorting detection task.

    Determines if a sequence of integers is sorted in ascending order.

    Loads data from JSONL files with format:
        {"sequence": [1, 3, 5, 7, 9], "is_sorted": 1, "length": 5}
    """

    def __init__(self, data_path):
        """
        Args:
            data_path: Path to JSONL file
        """
        self.samples = []

        # Load data
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            sequence: Tensor of integers [seq_len]
            label: Binary label (1 = sorted, 0 = unsorted)
            length: Sequence length (for analysis)
        """
        sample = self.samples[idx]
        sequence = torch.tensor(sample['sequence'], dtype=torch.long)
        label = torch.tensor(sample['is_sorted'], dtype=torch.long)
        length = sample['length']

        return sequence, label, length


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.

    Pads all sequences in batch to the same length.

    Args:
        batch: List of (sequence, label, length) tuples

    Returns:
        sequences_padded: Padded sequences [batch, max_len]
        labels: Binary labels [batch]
        lengths: Original sequence lengths [batch]
    """
    sequences, labels, lengths = zip(*batch)

    # Find max length in batch
    max_len = max(len(seq) for seq in sequences)

    # Pad sequences (pad with 0 which is smallest value)
    sequences_padded = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with zeros (won't affect sorting since sequences use values 0-99)
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            seq_padded = torch.cat([seq, padding])
        else:
            seq_padded = seq
        sequences_padded.append(seq_padded)

    sequences_padded = torch.stack(sequences_padded)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return sequences_padded, labels, lengths


def create_dataloaders(data_dir, batch_size=128, num_workers=0):
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing train.jsonl and val.jsonl
        batch_size: Batch size
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, val_loader
    """
    import os

    train_dataset = SortingDetectionDataset(os.path.join(data_dir, 'train.jsonl'))
    val_dataset = SortingDetectionDataset(os.path.join(data_dir, 'val.jsonl'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def create_test_dataloader(data_path, batch_size=128, num_workers=0):
    """
    Create test dataloader for a specific test length.

    Args:
        data_path: Path to test JSONL file (e.g., test_len_32.jsonl)
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        test_loader
    """
    test_dataset = SortingDetectionDataset(data_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return test_loader


# ============================================================================
# Test code
# ============================================================================

if __name__ == '__main__':
    import os

    data_dir = '../data/problem2'
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Run: make data")
        exit(1)

    print("Testing dataset loading...")

    # Test train/val loaders
    train_loader, val_loader = create_dataloaders(data_dir, batch_size=4)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Get one batch
    sequences, labels, lengths = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Lengths: {lengths.shape}")

    # Show examples
    print(f"\nExample sequences:")
    for i in range(min(3, len(sequences))):
        seq = sequences[i, :lengths[i]].tolist()
        label = labels[i].item()
        is_actually_sorted = seq == sorted(seq)
        print(f"  {i+1}. Length {lengths[i]}: {seq}")
        print(f"     Label: {label} (sorted={label==1}), "
              f"Actually sorted: {is_actually_sorted}, "
              f"Match: {(label==1) == is_actually_sorted}")

    # Test length distribution
    print(f"\nLength distribution in training set:")
    length_counts = {}
    for _, _, lengths_batch in train_loader:
        for length in lengths_batch.tolist():
            length_counts[length] = length_counts.get(length, 0) + 1

    for length in sorted(length_counts.keys()):
        print(f"  Length {length:3d}: {length_counts[length]:4d} samples")

    # Test extrapolation datasets
    print(f"\nTesting extrapolation datasets:")
    test_lengths = [32, 64, 128, 256]
    for test_len in test_lengths:
        test_path = os.path.join(data_dir, f'test_len_{test_len}.jsonl')
        if os.path.exists(test_path):
            test_loader = create_test_dataloader(test_path, batch_size=4)
            print(f"  Length {test_len}: {len(test_loader)} batches")

            # Verify all sequences are correct length
            sequences, labels, lengths = next(iter(test_loader))
            assert all(l == test_len for l in lengths.tolist()), \
                f"Found incorrect lengths in test_len_{test_len}"
        else:
            print(f"  Length {test_len}: File not found")

    print("\n✓ Dataset tests passed!")
