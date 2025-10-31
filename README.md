# EE 641 - HW3 Starter Code

## Assignment: Attention Mechanisms and Transformers

This starter code provides the structure for implementing multi-head attention and exploring positional encoding strategies.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate datasets
python problem1/scripts/generate_data.py
python problem2/scripts/generate_data.py

# 3. Verify your implementations
python problem1/utils/verify_attention.py
python problem2/utils/verify_encoding.py

# 4. Train models
cd problem1 && python train.py
cd problem2 && python train.py
```

## Structure

```
hw3-starter/
├── problem1/               # Multi-Head Attention
│   ├── src/
│   │   ├── attention.py   # TODO: Implement attention mechanism
│   │   ├── dataset.py     # TODO: Implement collate function
│   │   ├── model.py       # PROVIDED: Transformer architecture
│   │   └── __init__.py
│   ├── scripts/
│   │   └── generate_data.py  # PROVIDED: Data generation
│   ├── utils/
│   │   └── verify_attention.py  # PROVIDED: Test your attention
│   └── train.py           # PARTIAL: Implement loss masking & accuracy
│
├── problem2/               # Positional Encoding
│   ├── src/
│   │   ├── positional_encoding.py  # TODO: Implement 3 encodings
│   │   ├── dataset.py     # PROVIDED: Sorting dataset
│   │   ├── model.py       # PROVIDED: Transformer encoder
│   │   └── __init__.py
│   ├── scripts/
│   │   └── generate_data.py  # PROVIDED: Data generation
│   ├── utils/
│   │   └── verify_encoding.py  # PROVIDED: Test your encodings
│   └── train.py           # PROVIDED: Training script
│
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## What You Need to Implement

### Problem 1: Multi-Head Attention (~150 lines)
- `src/attention.py`:
  - `scaled_dot_product_attention()` - Core attention mechanism
  - `MultiHeadAttention` class - Split into multiple heads
- `src/dataset.py`:
  - `collate_fn_variable_length()` - Handle padding
- `train.py`:
  - `compute_masked_loss()` - Loss computation ignoring padding
  - `calculate_accuracy()` - Token and sequence accuracy

### Problem 2: Positional Encoding (~100 lines)
- `src/positional_encoding.py`:
  - `SinusoidalPositionalEncoding` - Fixed trigonometric encoding
  - `LearnedPositionalEncoding` - Trainable embeddings
  - `NoPositionalEncoding` - Baseline without position info

## Development Workflow

1. **Start with verification scripts** - They show you what's expected
2. **Implement incrementally** - Test each function as you write it
3. **Use the test code** - Every file has `if __name__ == '__main__'` tests
4. **Read the TODOs** - They guide you through the implementation
5. **Run verification** - Make sure everything works before training

## Testing Your Implementation

### Before Training
```bash
# Test attention implementation
python problem1/utils/verify_attention.py

# Test positional encodings
python problem2/utils/verify_encoding.py
```

### Individual Components
```bash
# Test attention module
python problem1/src/attention.py

# Test dataset loading
python problem1/src/dataset.py

# Test model architecture
python problem1/src/model.py

# Test positional encodings
python problem2/src/positional_encoding.py
```

## Common Issues

### Import Errors
Make sure you're running scripts from the correct directory:
```bash
cd problem1
python train.py  # Not: python problem1/train.py
```

### Shape Mismatches
Check the comments - they specify expected tensor shapes:
```python
# Q: [batch, num_heads, seq_len, d_k]
```

### NotImplementedError
You haven't implemented that function yet - look for the TODO comment.

### Data Not Found
Generate the datasets first:
```bash
python problem1/scripts/generate_data.py
python problem2/scripts/generate_data.py
```

## Getting Help

1. **Read the docstrings** - They explain what each function should do
2. **Check the hints** - TODO comments include implementation hints
3. **Run the tests** - They show you what's failing
4. **Review the assignment page** - See the full problem description

## Expected Time

- Problem 1: 5-7 hours (attention + training + analysis)
- Problem 2: 3-5 hours (encodings + extrapolation analysis)
- Total: 10-14 hours

## Tips

- **Start simple**: Get basic versions working before optimizing
- **Test frequently**: Don't write everything before testing
- **Read the code**: The provided files show you how components fit together
- **Use verification**: The verify scripts catch most bugs before training
- **Check shapes**: Most bugs are tensor shape mismatches

Good luck!
