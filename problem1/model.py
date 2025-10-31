"""
Sequence-to-sequence transformer model.

PROVIDED COMPLETE - Uses your MultiHeadAttention implementation from attention.py.

This file demonstrates how the attention mechanism you implement is used in a
full transformer architecture. You don't need to modify this file, but you
should understand how the components fit together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.attention import MultiHeadAttention, create_causal_mask


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention is All You Need'.

    Adds position information to embeddings using sine and cosine functions.
    This allows the model to use the order of the sequence.

    Note: You'll explore different positional encoding strategies in Problem 2.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied to each position independently and identically.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Structure: Self-Attention -> Add & Norm -> FFN -> Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)  # Your implementation!
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output [batch, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    Structure: Masked Self-Attention -> Add & Norm -> Cross-Attention -> Add & Norm -> FFN -> Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)  # Your implementation!
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # Your implementation!
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention (causal)

        Returns:
            Output [batch, tgt_len, d_model]
        """
        # Masked self-attention (can't look ahead)
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention to encoder (attend to source sequence)
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)

        return x


class Seq2SeqTransformer(nn.Module):
    """
    Full sequence-to-sequence transformer.

    Uses your MultiHeadAttention implementation in both encoder and decoder.
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings (convert token IDs to vectors)
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection (convert back to vocabulary probabilities)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """
        Encode source sequence.

        Args:
            src: Source token IDs [batch, src_len]
            src_mask: Optional mask

        Returns:
            Encoder output [batch, src_len, d_model]
        """
        # Embed tokens and add positional encoding
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence.

        Args:
            tgt: Target token IDs [batch, tgt_len]
            encoder_output: Encoder output [batch, src_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Causal mask for self-attention

        Returns:
            Decoder output [batch, tgt_len, d_model]
        """
        # Embed tokens and add positional encoding
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass.

        Args:
            src: Source sequence [batch, src_len]
            tgt: Target sequence [batch, tgt_len]
            src_mask: Source mask
            tgt_mask: Target mask (causal)

        Returns:
            Logits [batch, tgt_len, vocab_size]
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits

    def generate(self, src, max_len, start_token=0, device='cpu'):
        """
        Generate sequence autoregressively using greedy decoding.

        Args:
            src: Source sequence [batch, src_len]
            max_len: Maximum generation length
            start_token: Start token ID
            device: Device to run on

        Returns:
            Generated sequence [batch, max_len]
        """
        self.eval()
        batch_size = src.size(0)

        with torch.no_grad():
            # Encode source once
            encoder_output = self.encode(src)

            # Start with start token
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

            # Generate one token at a time
            for _ in range(max_len - 1):
                # Create causal mask
                tgt_mask = create_causal_mask(tgt.size(1), device=device)

                # Decode
                decoder_output = self.decode(tgt, encoder_output, tgt_mask=tgt_mask)

                # Get next token (greedy)
                logits = self.output_projection(decoder_output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to sequence
                tgt = torch.cat([tgt, next_token], dim=1)

        return tgt


# ============================================================================
# Test code
# ============================================================================

if __name__ == '__main__':
    print("Testing Seq2SeqTransformer model...")
    print("=" * 60)

    vocab_size = 20
    d_model = 128
    num_heads = 4
    batch_size = 2
    src_len = 16
    tgt_len = 16

    try:
        # Create model
        model = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        print(f"✓ Model created successfully")
        print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        src = torch.randint(0, vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
        tgt_mask = create_causal_mask(tgt_len)

        logits = model(src, tgt, tgt_mask=tgt_mask)

        print(f"\n✓ Forward pass successful")
        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        print(f"  Output shape: {logits.shape}")

        # Test generation
        generated = model.generate(src, max_len=tgt_len, start_token=0)
        print(f"\n✓ Generation successful")
        print(f"  Generated shape: {generated.shape}")

        print("\n" + "=" * 60)
        print("All model tests passed!")
        print("=" * 60)
        print("\nThis means your attention implementation is compatible")
        print("with the full transformer architecture.")

    except NotImplementedError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you've implemented MultiHeadAttention in attention.py")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("\nCheck your attention implementation for bugs")
