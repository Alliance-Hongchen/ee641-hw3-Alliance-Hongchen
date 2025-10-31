import os
import json
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.dataset import create_test_dataloader
from src.model import create_model

# Configuration
DATA_DIR = '../data/problem2'
RESULTS_DIR = 'results'
EXTRAPOLATION_DIR = os.path.join(RESULTS_DIR, 'extrapolation')
ENCODING_TYPES = ['sinusoidal', 'learned', 'none']
TEST_LENGTHS = [32, 64, 128, 256]
MAX_TRAIN_LEN = 16  # Model was trained on sequences of length 8-16


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy on a given test dataloader.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for sequences, labels, _ in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            logits = model(sequences)

            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

    return total_correct / total_samples


def plot_extrapolation_curves(results, lengths, save_path):
    """
    Plot and save the extrapolation accuracy curves.
    """
    plt.figure(figsize=(10, 6))

    for encoding_type, accuracies in results.items():
        plt.plot(lengths, accuracies, label=encoding_type.capitalize(), marker='o', linewidth=2)

    plt.title('Model Extrapolation Performance by Position Encoding', fontsize=16, fontweight='bold')
    plt.xlabel('Sequence Length (Log Scale)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(lengths, labels=lengths)
    plt.ylim(0.4, 1.05)  # Start y-axis at 0.4 to better see failure (0.5) vs success (1.0)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess (0.5)')
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"\n✓ Saved extrapolation curves to {save_path}")


def plot_learned_embeddings(model_path, save_path):
    """
    Load the learned encoding model and visualize its position embeddings.
    """
    print(f"\nGenerating learned embedding visualization...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract position embedding weights
    # find the 'pos_encoding.position_embeddings.weight' key
    try:
        embeddings = checkpoint['model_state_dict']['pos_encoding.position_embeddings.weight']
    except KeyError:
        print(f"✗ Error: Could not find 'pos_encoding.position_embeddings.weight' in {model_path}")
        print("  Was this model trained with 'learned' encoding?")
        return

    embeddings = embeddings.detach().cpu().numpy()  # [max_len, d_model]

    # only trained on positions up to 16, plot the first 64 to see the difference between trained and untrained embeddings
    emb_to_plot = embeddings[:64, :].T  # [d_model, 64]

    plt.figure(figsize=(14, 7))
    sns.heatmap(emb_to_plot, cmap='viridis', cbar_kws={'label': 'Embedding Value'})

    # Add a vertical line to show the boundary of training data
    plt.axvline(x=MAX_TRAIN_LEN - 0.5, color='red', linestyle='--', linewidth=2,
                label=f'End of Train Data (pos={MAX_TRAIN_LEN - 1})')

    plt.title('Learned Positional Embeddings (First 64 Positions)', fontsize=16, fontweight='bold')
    plt.xlabel('Position Index', fontsize=12)
    plt.ylabel('Embedding Dimension', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"✓ Saved learned embedding visualization to {save_path}")


def main():
    """
    Main function to run evaluation and generate all deliverables.
    """
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(EXTRAPOLATION_DIR, exist_ok=True)

    all_results = {}

    for encoding_type in ENCODING_TYPES:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {encoding_type.upper()}")
        print(f"{'=' * 60}")

        model_path = os.path.join(RESULTS_DIR, encoding_type, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"✗ Warning: Model file not found at {model_path}. Skipping.")
            all_results[encoding_type] = [0.0] * len(TEST_LENGTHS)
            continue

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Load config from checkpoint
        config = checkpoint['config']

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

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        accuracies = []
        for length in TEST_LENGTHS:
            print(f"  Testing on length={length}...")
            test_file = os.path.join(DATA_DIR, f'test_len_{length}.jsonl')
            if not os.path.exists(test_file):
                print(f"    ✗ Warning: Test file not found at {test_file}. Skipping length.")
                accuracies.append(0.0)  # Append 0 for missing data
                continue

            # Create dataloader
            test_loader = create_test_dataloader(test_file, batch_size=config['batch_size'])

            # Evaluate
            try:
                acc = evaluate_model(model, test_loader, device)
                accuracies.append(acc)
                print(f"    ✓ Accuracy: {acc:.4f}")
            except AssertionError as e:
                # This catches the 'LearnedPositionalEncoding' failure
                print(f"    ✗ Failed (as expected for learned): {e}")
                accuracies.append(0.0)  # Append 0 for failure
            except Exception as e:
                print(f"    ✗ An unexpected error occurred: {e}")
                accuracies.append(0.0)

        all_results[encoding_type] = accuracies

    # --- Quantitative Summary ---
    print("\n" + "=" * 60)
    print("Quantitative Extrapolation Summary")
    print("=" * 60)
    header = f"{'Encoding':12s} |" + "".join([f" Len {l:<5} |" for l in TEST_LENGTHS])
    print(header)
    print("-" * len(header))
    for encoding_type, accuracies in all_results.items():
        acc_str = "".join([f" {acc:7.4f} |" for acc in accuracies])
        print(f"{encoding_type:12s} |" + acc_str)
    print("=" * 60)

    # --- Save Deliverables ---

    # Save JSON results
    json_path = os.path.join(EXTRAPOLATION_DIR, 'extrapolation_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved JSON results to {json_path}")

    # Save extrapolation curves plot
    plot_path = os.path.join(EXTRAPOLATION_DIR, 'extrapolation_curves.png')
    plot_extrapolation_curves(all_results, TEST_LENGTHS, plot_path)

    # Save learned embeddings plot
    learned_model_path = os.path.join(RESULTS_DIR, 'learned', 'best_model.pth')
    if os.path.exists(learned_model_path):
        emb_plot_path = os.path.join(EXTRAPOLATION_DIR, 'learned_position_embeddings.png')
        plot_learned_embeddings(learned_model_path, emb_plot_path)
    else:
        print("\nSkipping learned embedding plot: 'results/learned/best_model.pth' not found.")

    print("\nEvaluation complete. All deliverables saved to results/extrapolation/")


if __name__ == '__main__':
    main()