import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import List, Optional


def plot_loss_perplexity(
    history: dict,
    save_path: Optional[str] = "saved_models/training_curves.png"
):
    """
    Plot training loss and perplexity curves across epochs and save figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], 'o-', color='#3b82f6', linewidth=2, label='Training Loss')
    ax1.set_title('Seq2Seq Model Loss Over Epochs', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Perplexity plot
    ax2.plot(epochs, history["train_ppl"], 's-', color='#10b981', linewidth=2, label='Perplexity (PPL)')
    ax2.set_title('Model Perplexity Over Epochs', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Perplexity = exp(Loss)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_attention_heatmap(
    input_tokens: List[str],
    output_tokens: List[str],
    attention_matrix: List[List[float]],
    save_path: Optional[str] = None
):
    """
    Plot attention weight alignment heatmap between input sequence and output sequence.
    """
    attention_arr = np.array(attention_matrix)  # [trg_len, src_len]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_arr, cmap='Blues')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_tokens, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels([''] + output_tokens, fontsize=10)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Input Tokens (Encoder)', fontsize=11, labelpad=10)
    ax.set_ylabel('Generated Tokens (Decoder)', fontsize=11, labelpad=10)
    ax.set_title('Bahdanau Attention Alignment Matrix', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
