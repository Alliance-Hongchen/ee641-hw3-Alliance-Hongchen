"""
Positional encoding implementations.

STUDENT IMPLEMENTATION REQUIRED:
- SinusoidalPositionalEncoding - Fixed trigonometric encoding
- LearnedPositionalEncoding - Trainable position embeddings
- NoPositionalEncoding - Baseline without position information

This problem explores how different positional encoding strategies affect
the model's ability to generalize to longer sequences than seen during training.
"""

import torch
import torch.nn as nn
import math


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - Sinusoidal Positional Encoding
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention is All You Need'.

    Uses sine and cosine functions of different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Key property: This encoding can extrapolate to sequence lengths longer
    than those seen during training because it's defined by a mathematical
    formula, not learned parameters.

    Why it works:
        - Each dimension has a different frequency
        - Positions can be represented as linear combinations
        - The encoding is deterministic and continuous
        - Relative positions can be computed via dot products
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Dimension of model embeddings (must be even)
            max_len: Maximum sequence length to precompute encodings for

        You need to:
            1. Create a matrix pe of shape [max_len, d_model]
            2. Fill it with sinusoidal values according to the formula
            3. Register it as a buffer (not a parameter)
        """
        super().__init__()

        self.d_model = d_model

        # TODO: Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # TODO: Create position indices [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # TODO: Compute division term for frequency
        # Formula: div_term[i] = 1 / 10000^(2i/d_model)
        # Hint: Use exp and log to compute this stably
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # TODO: Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # TODO: Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # TODO: Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

        # raise NotImplementedError("Implement SinusoidalPositionalEncoding.__init__")

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional encoding [batch, seq_len, d_model]

        The positional encoding is added to the input, so the model
        receives both content (from embeddings) and position information.
        """
        # TODO: Get sequence length
        seq_len = x.size(1)

        # TODO: Add positional encoding (it broadcasts over batch dimension)
        # Check if the sequence length is within our precomputed buffer
        if seq_len <= self.pe.shape[1]:
            # TODO: Add positional encoding (it broadcasts over batch dimension)
            # Ensure the buffer is on the same device as the input x
            return x + self.pe[:, :seq_len, :].to(x.device)
        else:
            # Handle extrapolation: dynamically compute PE for seq_len > max_len
            # This is the whole point of sinusoidal encoding's extrapolation

            # Re-compute the division term (ensure it's on the correct device)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float().to(x.device) * (-math.log(10000.0) / self.d_model)
            )

            # Create position indices [seq_len, 1]
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)

            # Create PE matrix [1, seq_len, d_model]
            pe_long = torch.zeros(1, seq_len, self.d_model, device=x.device)

            # Apply sine to even indices
            pe_long[0, :, 0::2] = torch.sin(position * div_term)

            # Apply cosine to odd indices
            pe_long[0, :, 1::2] = torch.cos(position * div_term)

            return x + pe_long

        # raise NotImplementedError("Implement SinusoidalPositionalEncoding.forward")


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - Learned Positional Encoding
# ============================================================================

class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional embeddings.

    Each position gets a learnable embedding vector, similar to token embeddings.

    Key limitation: Unlike sinusoidal encoding, this CANNOT extrapolate to
    sequence lengths longer than max_len because positions beyond max_len
    don't have learned embeddings.

    Why it might be preferred during training:
        - Can learn task-specific position representations
        - Often works well when sequence lengths are fixed
        - Simpler to implement than relative positional encoding

    Why it fails at extrapolation:
        - No embeddings exist for positions > max_len
        - Even if we could access them, they weren't trained
        - The model has never seen how to process longer sequences
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize learned positional encoding.

        Args:
            d_model: Dimension of model embeddings
            max_len: Maximum sequence length (hard limit!)

        You need to:
            1. Create an Embedding layer with max_len positions
            2. Initialize the embeddings (use normal distribution)
        """
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # TODO: Create learnable position embeddings
        # This is an Embedding layer: one vector per position
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # TODO: Initialize with normal distribution
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.02)

        # raise NotImplementedError("Implement LearnedPositionalEncoding.__init__")

    def forward(self, x):
        """
        Add learned positional embeddings to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional embeddings [batch, seq_len, d_model]

        This will raise an assertion error if seq_len > max_len,
        demonstrating the limitation of learned encodings.
        """
        # TODO: Get dimensions
        batch_size, seq_len, d_model = x.size()

        # TODO: Ensure sequence length doesn't exceed max_len
        assert seq_len <= self.max_len, \
            f"Sequence length {seq_len} exceeds max_len {self.max_len}"

        # TODO: Create position indices [seq_len]
        positions = torch.arange(seq_len, device=x.device)

        # TODO: Get position embeddings [seq_len, d_model]
        pos_embeddings = self.position_embeddings(positions)

        # TODO: Add to input (broadcasts over batch dimension)
        return x + pos_embeddings.unsqueeze(0)

        # raise NotImplementedError("Implement LearnedPositionalEncoding.forward")


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - No Positional Encoding (Baseline)
# ============================================================================

class NoPositionalEncoding(nn.Module):
    """
    Baseline: No positional encoding.

    The model is permutation-invariant without positional information.
    It cannot distinguish between [1, 2, 3] and [3, 2, 1].

    This should fail on position-dependent tasks like:
        - Sorting detection
        - Sequence order prediction
        - Arithmetic (where digit position matters)

    Use this as a baseline to demonstrate that positional encoding
    is necessary for position-dependent tasks.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize (nothing to initialize for no-op encoding).

        Args:
            d_model: Dimension (unused, for interface compatibility)
            max_len: Maximum length (unused, for interface compatibility)
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        Return input unchanged (no positional information added).

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x (unchanged) [batch, seq_len, d_model]
        """
        # TODO: Just return x unchanged
        return x
        # raise NotImplementedError("Implement NoPositionalEncoding.forward")


# ============================================================================
# Utility function (PROVIDED)
# ============================================================================

def get_positional_encoding(encoding_type, d_model, max_len=5000):
    """
    Factory function to create positional encoding module.

    PROVIDED COMPLETE - Uses your implementations above.

    Args:
        encoding_type: One of 'sinusoidal', 'learned', 'none'
        d_model: Model dimension
        max_len: Maximum sequence length

    Returns:
        Positional encoding module
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model, max_len)
    elif encoding_type == 'learned':
        return LearnedPositionalEncoding(d_model, max_len)
    elif encoding_type == 'none':
        return NoPositionalEncoding(d_model, max_len)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


# ============================================================================
# Test code (for development and debugging)
# ============================================================================

if __name__ == '__main__':
    print("Testing positional encoding implementations...")
    print("=" * 60)

    batch_size, seq_len, d_model = 2, 16, 64
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")
    print(f"Input mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")

    # Test 1: Sinusoidal encoding
    print("\n1. Testing SinusoidalPositionalEncoding:")
    print("-" * 60)
    try:
        sin_pe = SinusoidalPositionalEncoding(d_model, max_len=1000)
        x_sin = sin_pe(x)
        print(f"✓ Output shape: {x_sin.shape}")
        print(f"✓ PE buffer shape: {sin_pe.pe.shape}")

        # Check that different positions have different encodings
        pe_pos0 = sin_pe.pe[0, 0, :]
        pe_pos1 = sin_pe.pe[0, 1, :]
        print(f"✓ Position 0 encoding (first 4): {pe_pos0[:4].tolist()}")
        print(f"✓ Position 1 encoding (first 4): {pe_pos1[:4].tolist()}")
        print(f"✓ L2 distance: {torch.norm(pe_pos0 - pe_pos1).item():.4f}")

        # Test extrapolation
        x_long = torch.randn(1, 128, d_model)
        x_long_sin = sin_pe(x_long)
        print(f"✓ Extrapolation test (len=128): {x_long_sin.shape}")

    except NotImplementedError:
        print("✗ Not yet implemented")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Learned encoding
    print("\n2. Testing LearnedPositionalEncoding:")
    print("-" * 60)
    try:
        learned_pe = LearnedPositionalEncoding(d_model, max_len=100)
        x_learned = learned_pe(x)
        print(f"✓ Output shape: {x_learned.shape}")
        print(f"✓ Embedding weight shape: {learned_pe.position_embeddings.weight.shape}")
        print(f"✓ Parameters: {sum(p.numel() for p in learned_pe.parameters()):,}")

        # Test that it fails on sequences longer than max_len
        print("\n  Testing failure on long sequences:")
        try:
            x_too_long = torch.randn(1, 150, d_model)
            x_too_long_learned = learned_pe(x_too_long)
            print("  ✗ Should have raised an assertion error!")
        except AssertionError as e:
            print(f"  ✓ Correctly raised error: {str(e)[:50]}...")

    except NotImplementedError:
        print("✗ Not yet implemented")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: No positional encoding
    print("\n3. Testing NoPositionalEncoding:")
    print("-" * 60)
    try:
        no_pe = NoPositionalEncoding(d_model)
        x_no_pe = no_pe(x)
        print(f"✓ Output shape: {x_no_pe.shape}")
        print(f"✓ Output equals input: {torch.allclose(x, x_no_pe)}")
        print(f"✓ Parameters: {sum(p.numel() for p in no_pe.parameters())}")

    except NotImplementedError:
        print("✗ Not yet implemented")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 4: Factory function
    print("\n4. Testing factory function:")
    print("-" * 60)
    for enc_type in ['sinusoidal', 'learned', 'none']:
        try:
            pe = get_positional_encoding(enc_type, d_model, max_len=200)
            x_out = pe(x)
            params = sum(p.numel() for p in pe.parameters())
            print(f"✓ {enc_type:12s}: shape {x_out.shape}, params {params:6d}")
        except NotImplementedError:
            print(f"✗ {enc_type:12s}: Not implemented")
        except Exception as e:
            print(f"✗ {enc_type:12s}: Error - {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
