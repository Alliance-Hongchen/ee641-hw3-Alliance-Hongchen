"""
Transformer encoder for sorting detection task.

Architecture PROVIDED to students (uses their positional encoding implementation).
Students import their positional encoding from positional_encoding.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoder(nn.Module):
    """
    Simple transformer encoder for binary classification.

    Uses student's positional encoding implementation from positional_encoding.py.
    """

    def __init__(
        self,
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_layers=4,
        d_ff=256,
        dropout=0.1,
        pos_encoding=None,
        max_seq_len=512
    ):
        """
        Args:
            vocab_size: Size of input vocabulary (for integers 0-99)
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            pos_encoding: Positional encoding module (from student implementation)
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (provided by student)
        self.pos_encoding = pos_encoding

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Input shape: [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # Binary classification
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input sequence [batch, seq_len] (integer tokens)

        Returns:
            logits: Classification logits [batch, 2]
        """
        # Embed tokens: [batch, seq_len] -> [batch, seq_len, d_model]
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Pass through transformer encoder
        # Output: [batch, seq_len, d_model]
        x = self.transformer_encoder(x)

        # Global average pooling over sequence dimension
        # [batch, seq_len, d_model] -> [batch, d_model]
        x = x.mean(dim=1)

        # Classification
        # [batch, d_model] -> [batch, 2]
        logits = self.classifier(x)

        return logits


def create_model(encoding_type, vocab_size=100, d_model=64, num_heads=4,
                 num_layers=4, d_ff=256, dropout=0.1, max_seq_len=512):
    """
    Factory function to create model with specified positional encoding.

    Args:
        encoding_type: One of 'sinusoidal', 'learned', 'none'
        vocab_size: Size of input vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length

    Returns:
        TransformerEncoder model
    """
    from src.positional_encoding import get_positional_encoding

    # Create positional encoding module
    pos_encoding = get_positional_encoding(encoding_type, d_model, max_seq_len)

    # Create model
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        pos_encoding=pos_encoding,
        max_seq_len=max_seq_len
    )

    return model


# ============================================================================
# Test code
# ============================================================================

if __name__ == '__main__':
    print("Testing TransformerEncoder model...")

    # Model parameters
    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 4
    batch_size = 2
    seq_len = 16

    # Test all three encoding types
    print("\nTesting all encoding types:")
    for encoding_type in ['sinusoidal', 'learned', 'none']:
        print(f"\n{encoding_type.upper()} Encoding:")

        model = create_model(
            encoding_type=encoding_type,
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )

        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Positional encoding parameters: "
              f"{sum(p.numel() for p in model.pos_encoding.parameters()):,}")

        # Create sample input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Expected: [batch={batch_size}, num_classes=2]")

        # Check output probabilities
        probs = F.softmax(logits, dim=-1)
        print(f"  Output probabilities (first sample): {probs[0].tolist()}")

    # Test extrapolation with different sequence lengths
    print("\n\nTesting different sequence lengths:")
    model_sin = create_model('sinusoidal', vocab_size=vocab_size, d_model=d_model,
                              num_heads=num_heads, num_layers=num_layers)
    model_learned = create_model('learned', vocab_size=vocab_size, d_model=d_model,
                                  num_heads=num_heads, num_layers=num_layers,
                                  max_seq_len=20)

    for seq_len in [8, 16, 32]:
        print(f"\nSequence length: {seq_len}")
        x = torch.randint(0, vocab_size, (1, seq_len))

        # Sinusoidal should work for all lengths
        try:
            logits_sin = model_sin(x)
            print(f"  Sinusoidal: ✓ Output shape {logits_sin.shape}")
        except Exception as e:
            print(f"  Sinusoidal: ✗ {e}")

        # Learned should fail for seq_len > max_seq_len
        try:
            logits_learned = model_learned(x)
            if seq_len <= 20:
                print(f"  Learned: ✓ Output shape {logits_learned.shape}")
            else:
                print(f"  Learned: ✗ Should have failed (seq_len > max_seq_len)")
        except Exception as e:
            if seq_len > 20:
                print(f"  Learned: ✓ Correctly failed for long sequence")
            else:
                print(f"  Learned: ✗ Unexpected error: {e}")

    print("\n✓ Model tests passed!")
