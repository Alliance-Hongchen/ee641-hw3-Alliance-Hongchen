import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

import sys

sys.path.insert(0, os.path.abspath('.'))
from src.model import Seq2SeqTransformer
from src.attention import MultiHeadAttention, create_causal_mask
from src.dataset import create_dataloaders
from train import evaluate  # re-use the evaluate function

# Global storage for hooks
attention_weights_storage = {}


def get_attention_hook(layer_name):
    """
    Returns a hook function that captures the output (attention weights) of a MultiHeadAttention module.
    """

    def hook(module, input, output):
        # The output of our MHA is (attn_output, attention_weights)
        attention_weights_storage[layer_name] = output[1].detach().cpu()

    return hook


def plot_attention_heatmaps(storage, src_tokens, tgt_tokens, save_dir, sample_idx=0):
    """
    Generates and saves heatmaps for all captured attention weights.
    """
    print(f"--- Plotting heatmaps for sample {sample_idx} ---")

    # Helper to convert tokens to strings (10 -> '+')
    def tokens_to_str(tokens):
        return [str(t) if t != 10 else '+' for t in tokens]

    src_labels = tokens_to_str(src_tokens[sample_idx].tolist())

    # Decoder target is shifted, so use tgt[:, 1:]
    tgt_labels = tokens_to_str(tgt_tokens[sample_idx, 1:].tolist())
    # Decoder input is shifted, so use tgt[:, :-1]
    tgt_in_labels = tokens_to_str(tgt_tokens[sample_idx, :-1].tolist())

    for layer_name, weights in storage.items():
        # Get weights for the specific sample: [num_heads, seq_len_q, seq_len_k]
        weights_sample = weights[sample_idx]
        num_heads = weights_sample.size(0)

        # Determine labels based on layer type
        x_labels, y_labels = None, None
        if 'encoder' in layer_name:
            # Encoder Self-Attention
            x_labels, y_labels = src_labels, src_labels
        elif 'decoder' in layer_name and 'cross' in layer_name:
            # Decoder Cross-Attention
            x_labels, y_labels = src_labels, tgt_in_labels
        elif 'decoder' in layer_name and 'self' in layer_name:
            # Decoder Self-Attention
            x_labels, y_labels = tgt_in_labels, tgt_in_labels

        fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 4, 4), squeeze=False)

        for h in range(num_heads):
            ax = axes[0, h]
            sns.heatmap(
                weights_sample[h],
                ax=ax,
                cmap='viridis',
                cbar=False,
                xticklabels=x_labels,
                yticklabels=y_labels
            )
            ax.set_title(f"Head {h}")
            ax.tick_params(axis='x', rotation=90)
            ax.tick_params(axis='y', rotation=0)

        fig.suptitle(f"Layer: {layer_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(save_dir, f"{layer_name.replace('.', '_')}.png")
        plt.savefig(save_path)
        plt.close(fig)
    print(f"Heatmaps saved to {save_dir}")


def run_head_ablation(baseline_model, config, test_loader, device):
    """
    Performs head ablation by systematically zeroing out parameters for each head in each MHA layer and re-evaluating.

    Returns:
        dict: A dictionary mapping (layer, head) to its performance.
    """
    print("--- Running Head Ablation Study ---")

    # Get baseline accuracy
    baseline_metrics = evaluate(baseline_model, test_loader, device, config['pad_idx'])
    baseline_acc = baseline_metrics[2]  # (loss, token_acc, seq_acc)
    print(f"Baseline Sequence Accuracy: {baseline_acc:.4f}")

    results = {}
    d_model = config['d_model']
    num_heads = config['num_heads']
    d_k = d_model // num_heads

    # Find all MHA layers in the model
    mha_layers = []
    for name, module in baseline_model.named_modules():
        if isinstance(module, MultiHeadAttention):
            mha_layers.append(name)

    total_heads_to_test = len(mha_layers) * num_heads
    pbar = tqdm(total=total_heads_to_test, desc="Ablating heads")

    for layer_name in mha_layers:
        for head_idx in range(num_heads):
            # Create a fresh copy of the model
            model_copy = Seq2SeqTransformer(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                num_encoder_layers=config['num_encoder_layers'],
                num_decoder_layers=config['num_decoder_layers'],
                d_ff=config['d_ff']
            ).to(device)
            # Load the original weights
            model_copy.load_state_dict(baseline_model.state_dict())
            model_copy.eval()

            # Get its state_dict for "surgery"
            state_dict = model_copy.state_dict()

            # Define parameters to zero out
            # These are the projections for Q, K, V for this head
            q_weight = f'{layer_name}.W_q.weight'
            k_weight = f'{layer_name}.W_k.weight'
            v_weight = f'{layer_name}.W_v.weight'
            q_bias = f'{layer_name}.W_q.bias'
            k_bias = f'{layer_name}.W_k.bias'
            v_bias = f'{layer_name}.W_v.bias'
            # This is the output projection
            o_weight = f'{layer_name}.W_o.weight'

            # Perform "surgery": zero out the weights for this head
            start_row = head_idx * d_k
            end_row = (head_idx + 1) * d_k

            # Zero Q, K, V projection weights (output dimension)
            state_dict[q_weight][start_row:end_row, :] = 0
            state_dict[k_weight][start_row:end_row, :] = 0
            state_dict[v_weight][start_row:end_row, :] = 0

            # Zero Q, K, V biases
            state_dict[q_bias][start_row:end_row] = 0
            state_dict[k_bias][start_row:end_row] = 0
            state_dict[v_bias][start_row:end_row] = 0

            # Zero output projection weights (input dimension)
            state_dict[o_weight][:, start_row:end_row] = 0

            # Load the ablated state_dict back into the model
            model_copy.load_state_dict(state_dict)

            # Evaluate this ablated model
            _, _, ablated_seq_acc = evaluate(model_copy, test_loader, device, config['pad_idx'])

            # Store result
            key = f'{layer_name}.head_{head_idx}'
            results[key] = {
                'ablated_acc': ablated_seq_acc,
                'acc_drop': baseline_acc - ablated_seq_acc
            }
            pbar.update(1)

    pbar.close()
    results['baseline_acc'] = baseline_acc
    print("Head ablation study complete.")
    return results


def save_ablation_results(results, save_dir):
    """
    Saves ablation results to JSON and plots a summary bar chart.
    """
    # Save raw data
    json_path = os.path.join(save_dir, 'head_ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results saved to {json_path}")

    # --- Create Importance Ranking ---
    baseline_acc = results.pop('baseline_acc')
    # Sort heads by accuracy drop (most drop = most important)
    sorted_heads = sorted(
        results.items(),
        key=lambda item: item[1]['acc_drop'],
        reverse=True
    )

    ranking_path = os.path.join(save_dir, 'head_importance_ranking.txt')
    with open(ranking_path, 'w') as f:
        f.write("Head Importance Ranking (Most Important First)\n")
        f.write("=" * 40 + "\n")
        f.write(f"Baseline Sequence Accuracy: {baseline_acc:.4f}\n\n")

        prunable_count = 0
        for i, (name, data) in enumerate(sorted_heads):
            drop = data['acc_drop']
            f.write(f"{i + 1:2d}. {name:40s} | Acc Drop: {drop:+.4f}\n")
            # Define "minimal loss" as < 1% absolute accuracy drop
            if abs(drop) < 0.01:
                prunable_count += 1

    print(f"Head importance ranking saved to {ranking_path}")

    # --- Plot Bar Chart ---
    labels = list(results.keys())
    acc_drops = [d['acc_drop'] for d in results.values()]

    plt.figure(figsize=(max(15, len(labels) * 0.5), 8))
    plt.bar(labels, acc_drops)
    plt.xticks(rotation=90)
    plt.ylabel('Accuracy Drop (Higher = More Important)')
    plt.title('Head Ablation Study: Impact of Pruning Each Head')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'head_ablation_summary.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Ablation summary plot saved to {plot_path}")

    # --- Report quantitative pruning results ---
    total_heads = len(labels)
    prunable_percent = (prunable_count / total_heads) * 100
    print("\n--- Quantitative Pruning Analysis ---")
    print(f"Total heads: {total_heads}")
    print(f"Heads prunable with < 1% accuracy loss: {prunable_count}")
    print(f"Percentage of heads that can be pruned: {prunable_percent:.2f}%")
    print("--------------------------------------")


def main():
    # --- Setup ---
    RESULTS_DIR = 'results'
    ATTN_DIR = os.path.join(RESULTS_DIR, 'attention_patterns')
    HEAD_ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'head_analysis')
    CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'best_model.pth')

    os.makedirs(ATTN_DIR, exist_ok=True)
    os.makedirs(HEAD_ANALYSIS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Model, Config, and Data ---
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found.")
        print("Please run train.py first to generate the model.")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint['config']

    # add any config keys that were not present at training time
    if 'pad_idx' not in config:
        config['pad_idx'] = 0  # Default from dataset.py

    print("Loading model...")
    model = Seq2SeqTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_ff=config['d_ff']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Loading data...")
    _, _, test_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size']
    )
    src, tgt = next(iter(test_loader))
    src, tgt = src.to(device), tgt.to(device)

    # PART 1: VISUALIZE ATTENTION PATTERNS
    global attention_weights_storage
    attention_weights_storage.clear()
    hook_handles = []

    print("Registering hooks to capture attention...")
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            handle = module.register_forward_hook(get_attention_hook(name))
            hook_handles.append(handle)

    # Run one forward pass to trigger the hooks and capture weights
    with torch.no_grad():
        tgt_input = tgt[:, :-1]
        tgt_mask = create_causal_mask(tgt_input.size(1), device=device)
        model(src, tgt_input, tgt_mask=tgt_mask)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    # Plot the captured weights
    plot_attention_heatmaps(
        attention_weights_storage,
        src.cpu(),
        tgt.cpu(),
        ATTN_DIR,
        sample_idx=0  # Plot for the first sample in the batch
    )

    # PART 2: RUN HEAD ABLATION STUDY
    # pass the original model (model)
    ablation_results = run_head_ablation(
        model,
        config,
        test_loader,
        device
    )

    save_ablation_results(ablation_results, HEAD_ANALYSIS_DIR)

    print("\n==================================")
    print("Analysis complete.")
    print(f"Visualizations saved to: {ATTN_DIR}")
    print(f"Ablation results saved to: {HEAD_ANALYSIS_DIR}")
    print("==================================")


if __name__ == '__main__':
    main()