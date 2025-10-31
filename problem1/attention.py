"""
Attention mechanisms implementation.

STUDENT IMPLEMENTATION REQUIRED:
- scaled_dot_product_attention()
- MultiHeadAttention class

This is the core of the assignment - implement attention from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - Scaled Dot-Product Attention
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    The attention mechanism computes a weighted sum of values (V) where
    the weights are determined by the similarity between queries (Q) and keys (K).

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query tensor [batch, ..., seq_len, d_k]
        K: Key tensor [batch, ..., seq_len, d_k]
        V: Value tensor [batch, ..., seq_len, d_k]
        mask: Optional mask [batch, ..., seq_len, seq_len]
              1 = attend, 0 = mask (will be converted to -inf before softmax)

    Returns:
        output: Attention output [batch, ..., seq_len, d_k]
        attention_weights: Attention weights [batch, ..., seq_len, seq_len]
                          (after softmax, these sum to 1 along last dimension)

    Implementation Steps:
        1. Compute attention scores: QK^T (matrix multiply Q with K transposed)
        2. Scale by sqrt(d_k) to prevent gradients from vanishing
        3. Apply mask if provided (set masked positions to -inf)
        4. Apply softmax to get attention weights (probabilities)
        5. Multiply attention weights by V to get output

    Hints:
        - Use torch.matmul() for matrix multiplication
        - Use .transpose(-2, -1) to swap last two dimensions
        - Use .masked_fill(mask == 0, float('-inf')) for masking
        - Q.size(-1) gives you d_k
    """
    # TODO: Get the dimension d_k from Q
    d_k = Q.size(-1)

    # TODO: Compute attention scores QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # TODO: Scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)

    # TODO: Apply mask if provided (set positions where mask == 0 to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # TODO: Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # TODO: Apply attention weights to values
    output = torch.matmul(attention_weights, V)

    # raise NotImplementedError("Implement scaled_dot_product_attention")

    return output, attention_weights


# ============================================================================
# TODO: STUDENT IMPLEMENTATION - Multi-Head Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Instead of performing a single attention function, multi-head attention
    projects Q, K, V to multiple different subspaces and performs attention
    in parallel. This allows the model to attend to information from different
    representation subspaces.

    Architecture:
        1. Linear projection: Split d_model into num_heads of dimension d_k
        2. Parallel attention: Apply attention to each head independently
        3. Concatenate: Combine all heads back together
        4. Output projection: Final linear transformation

    Why it works:
        - Different heads can learn to attend to different types of relationships
        - Increases model capacity without increasing sequence length complexity
        - Enables learning both local and global dependencies
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads

        The dimension per head is d_k = d_model // num_heads

        You need to create:
            - W_q, W_k, W_v: Linear projections for queries, keys, values
            - W_o: Output projection after concatenating heads
        """
        super().__init__()

        # TODO: Verify d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # TODO: Store dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Create linear projections for Q, K, V
        # Each projects from d_model -> d_model (we'll split into heads later)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # TODO: Create output projection
        self.W_o = nn.Linear(d_model, d_model)

        # raise NotImplementedError("Implement MultiHeadAttention.__init__")

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).

        This reshapes the tensor to have separate heads that can be processed
        in parallel.

        Args:
            x: Tensor of shape [batch, seq_len, d_model]

        Returns:
            Tensor of shape [batch, num_heads, seq_len, d_k]

        Implementation:
            1. Reshape: [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
            2. Transpose: [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        """
        # TODO: Get dimensions
        batch_size, seq_len, d_model = x.size()

        # TODO: Reshape to separate heads
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # TODO: Transpose to put heads in dimension 1
        x = x.transpose(1, 2)

        return x
        # raise NotImplementedError("Implement split_heads")

    def combine_heads(self, x):
        """
        Combine heads back into a single dimension.

        Inverse operation of split_heads.

        Args:
            x: Tensor of shape [batch, num_heads, seq_len, d_k]

        Returns:
            Tensor of shape [batch, seq_len, d_model]

        Implementation:
            1. Transpose: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, num_heads, d_k]
            2. Reshape: [batch, seq_len, num_heads, d_k] -> [batch, seq_len, d_model]
        """
        # TODO: Get dimensions
        batch_size, num_heads, seq_len, d_k = x.size()

        # TODO: Transpose heads back
        x = x.transpose(1, 2)

        # TODO: Reshape to combine heads (use .contiguous() before .view())
        x = x.contiguous().view(batch_size, seq_len, self.d_model)

        return x
        # raise NotImplementedError("Implement combine_heads")

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional mask [batch, 1, seq_len_q, seq_len_k] or broadcastable

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]

        Implementation steps:
            1. Apply linear projections to Q, K, V
            2. Split into multiple heads
            3. Apply scaled dot-product attention (call your function!)
            4. Combine heads back together
            5. Apply output projection
        """
        # TODO: Apply linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # TODO: Split into multiple heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len_q, d_k]
        K = self.split_heads(K)  # [batch, num_heads, seq_len_k, d_k]
        V = self.split_heads(V)  # [batch, num_heads, seq_len_v, d_k]

        # TODO: Apply scaled dot-product attention
        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # TODO: Combine heads
        attn_output = self.combine_heads(attn_output)  # [batch, seq_len_q, d_model]

        # TODO: Apply output projection
        output = self.W_o(attn_output)

        return output, attention_weights
        # raise NotImplementedError("Implement MultiHeadAttention.forward")


# ============================================================================
# Utility functions for masking (PROVIDED - you can use these)
# ============================================================================

def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask for a sequence.

    Args:
        seq: Sequence tensor [batch, seq_len]
        pad_idx: Padding token index

    Returns:
        Mask tensor [batch, 1, 1, seq_len] where 0 = padding, 1 = real token
    """
    # Create mask: 1 for real tokens, 0 for padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(seq_len, device=None):
    """
    Create causal (look-ahead) mask for decoder self-attention.

    Prevents position i from attending to positions j > i.
    This ensures the decoder can only look at previous positions.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] where 0 = masked, 1 = allowed

    Example for seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    # Create lower triangular matrix (1s below and on diagonal, 0s above)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # Add batch and head dimensions
    return mask.unsqueeze(0).unsqueeze(0)


# ============================================================================
# Test code (for development and debugging)
# ============================================================================

if __name__ == '__main__':
    print("Testing attention implementation...")
    print("=" * 60)

    # Test 1: Scaled dot-product attention
    print("\n1. Testing scaled_dot_product_attention:")
    print("-" * 60)
    batch_size, seq_len, d_k = 2, 4, 8
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    try:
        output, weights = scaled_dot_product_attention(Q, K, V)
        print(f"✓ Input shape: {Q.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Attention weights shape: {weights.shape}")
        print(f"✓ Weights sum to 1.0: {weights.sum(dim=-1)[0, 0]:.4f}")
        assert output.shape == Q.shape, "Output shape mismatch"
        assert weights.shape == (batch_size, seq_len, seq_len), "Weights shape mismatch"
    except NotImplementedError:
        print("✗ Not yet implemented")

    # Test 2: Attention with causal mask
    print("\n2. Testing with causal mask:")
    print("-" * 60)
    try:
        mask = create_causal_mask(seq_len)
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        print(f"✓ Masked attention weights shape: {weights.shape}")
        print(f"✓ First sequence attention pattern (should be lower triangular):")
        print(weights[0].detach().numpy().round(3))

        # Verify causal property: upper triangle should be zero
        upper_triangle = weights[0].triu(diagonal=1)
        assert upper_triangle.sum() < 1e-6, "Causal mask not working - found attention to future"
        print("✓ Causal masking working correctly")
    except NotImplementedError:
        print("✗ Not yet implemented")

    # Test 3: Multi-head attention
    print("\n3. Testing MultiHeadAttention:")
    print("-" * 60)
    d_model, num_heads = 128, 4
    try:
        mha = MultiHeadAttention(d_model, num_heads)

        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output, weights = mha(query, key, value)
        print(f"✓ Input shape: {query.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Attention weights shape: {weights.shape}")
        print(f"✓ Number of parameters: {sum(p.numel() for p in mha.parameters()):,}")

        assert output.shape == query.shape, "Output shape mismatch"
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len), "Weights shape mismatch"
    except NotImplementedError:
        print("✗ Not yet implemented")

    # Test 4: Self-attention (Q=K=V)
    print("\n4. Testing self-attention (Q=K=V):")
    print("-" * 60)
    try:
        x = torch.randn(batch_size, seq_len, d_model)
        output, weights = mha(x, x, x)
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        assert output.shape == x.shape, "Self-attention output shape mismatch"
    except NotImplementedError:
        print("✗ Not yet implemented")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
