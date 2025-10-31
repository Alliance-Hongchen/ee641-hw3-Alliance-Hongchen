"""
Dataset loader for multi-digit addition task.

PARTIALLY PROVIDED - You need to implement:
- collate_fn_variable_length() - Handle variable-length sequences with padding

The dataset is provided, but you must understand how to batch variable-length
sequences properly for the transformer.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SequenceDataset(Dataset):
    """
    Dataset for sequence-to-sequence tasks.

    Handles variable-length sequences with padding.

    Loads data from JSONL files with format:
        {"input": [1, 2, 3, 4], "target": [4, 3, 2, 1]}

    PROVIDED COMPLETE - No modification needed.
    """

    def __init__(self, data_path, vocab_size=20, pad_idx=0):
        """
        Args:
            data_path: Path to JSONL file
            vocab_size: Size of vocabulary (for validation)
            pad_idx: Padding token index (default: 0)
        """
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
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
            input_seq: Tensor of shape [seq_len]
            target_seq: Tensor of shape [seq_len]
        """
        sample = self.samples[idx]
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target_seq = torch.tensor(sample['target'], dtype=torch.long)
        return input_seq, target_seq


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - Collate Function
# ============================================================================

def collate_fn_variable_length(batch, pad_idx=0):
    """
    Custom collate function to handle variable-length sequences.

    When batching sequences of different lengths, we need to pad them to the
    same length so they can be stacked into a single tensor.

    Args:
        batch: List of (input_seq, target_seq) tuples from __getitem__
               Each sequence is a 1D tensor of varying length
        pad_idx: Padding token index (default: 0)

    Returns:
        src_padded: Padded input sequences [batch, max_src_len]
        tgt_padded: Padded target sequences [batch, max_tgt_len]

    Why this matters:
        - Transformers process batches in parallel for efficiency
        - All sequences in a batch must have the same shape
        - Padding allows variable-length sequences to be batched
        - The attention mask will handle ignoring padding positions

    Implementation hints:
        Option 1: Use torch.nn.utils.rnn.pad_sequence()
            - Takes a list of tensors
            - Returns padded tensor [batch, max_len] if batch_first=True
            - Set padding_value=pad_idx

        Option 2: Manual padding
            - Find max length in batch
            - Create zero tensor of shape [batch_size, max_len]
            - Copy each sequence into the tensor
    """
    # TODO: Separate inputs and targets
    inputs, targets = zip(*batch)

    # TODO: Pad sequences to max length in batch
    # Hint: pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    src_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    return src_padded, tgt_padded
    # raise NotImplementedError("Implement collate_fn_variable_length")


def create_dataloaders(data_dir, batch_size=64, num_workers=0):
    """
    Create train, val, and test dataloaders.

    PROVIDED COMPLETE - Uses your collate_fn_variable_length implementation.

    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        batch_size: Batch size
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    import os

    train_dataset = SequenceDataset(os.path.join(data_dir, 'train.jsonl'))
    val_dataset = SequenceDataset(os.path.join(data_dir, 'val.jsonl'))
    test_dataset = SequenceDataset(os.path.join(data_dir, 'test.jsonl'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_variable_length  # Uses your implementation!
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_variable_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_variable_length
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Test code
# ============================================================================

if __name__ == '__main__':
    # Test dataset loading
    import os

    data_dir = '../data/problem1'
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Run: python scripts/generate_data.py")
        exit(1)

    print("Testing dataset loading...")
    print("=" * 60)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size=4)

        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        print(f"✓ Test batches: {len(test_loader)}")

        # Get one batch
        input_seq, target_seq = next(iter(train_loader))
        print(f"\n✓ Batch shapes:")
        print(f"  Input: {input_seq.shape}")
        print(f"  Target: {target_seq.shape}")

        print(f"\n✓ Example sequences:")
        for i in range(min(3, input_seq.size(0))):
            print(f"  Input:  {input_seq[i].tolist()}")
            print(f"  Target: {target_seq[i].tolist()}")

        print("\n" + "=" * 60)
        print("Dataset loading successful!")
        print("=" * 60)

    except NotImplementedError:
        print("\n✗ collate_fn_variable_length not yet implemented")
        print("Implement the collate function to test data loading")
