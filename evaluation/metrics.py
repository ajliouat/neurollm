"""Evaluation metrics and attention visualization.

Provides:
  • per-subject accuracy, Cohen's κ, confusion matrix
  • attention map extraction from transformer layers
  • topographic attention plots (highlight C3/C4 for motor imagery)
  • publication-quality figure generation
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : (N,) int array
    y_pred : (N,) int array
    class_names : optional list of class labels

    Returns
    -------
    dict with accuracy, kappa, confusion_matrix, per_class_accuracy
    """
    if class_names is None:
        class_names = ["Left", "Right", "Feet", "Tongue"]

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[name] = float((y_pred[mask] == i).mean())
        else:
            per_class[name] = 0.0

    return {
        "accuracy": float(acc),
        "kappa": float(kappa),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class,
        "n_samples": len(y_true),
    }


def aggregate_subject_metrics(
    subject_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across subjects.

    Parameters
    ----------
    subject_metrics : list of metric dicts from compute_metrics

    Returns
    -------
    dict with mean_accuracy, std_accuracy, mean_kappa, per_subject results
    """
    accs = [m["accuracy"] for m in subject_metrics]
    kappas = [m["kappa"] for m in subject_metrics]

    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas)),
        "per_subject": subject_metrics,
        "n_subjects": len(subject_metrics),
    }


# ── Attention extraction ─────────────────────────────────────────────

def extract_attention_maps(
    model: torch.nn.Module,
    x: Tensor,
    mode: str = "finetune",
) -> List[Tensor]:
    """Extract attention weights from all transformer layers.

    Manually iterates encoder layers and calls ``self_attn`` with
    ``need_weights=True``, since PyTorch's ``TransformerEncoderLayer``
    hardcodes ``need_weights=False``.

    Parameters
    ----------
    model : NeuRoLLM
    x : Tensor (B, C, T)
    mode : str

    Returns
    -------
    attention_maps : list of Tensor, each (B, H, N, N)
    """
    model.eval()
    with torch.no_grad():
        B = x.shape[0]
        tokens = model.patch_embed(x)  # (B, N, d)
        N = tokens.shape[1]
        n_patches = N // model.n_channels

        pe = model.pos_enc(n_patches, x.device)
        tokens = tokens + pe

        if mode == "finetune":
            cls = model.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # Manually iterate through encoder layers
        attention_maps: List[Tensor] = []
        encoder_layers = model.encoder.encoder.layers

        hidden = tokens
        for layer in encoder_layers:
            # Pre-norm architecture: norm → self_attn → residual → norm → ffn → residual
            # TransformerEncoderLayer with norm_first=True does:
            #   x = x + _sa_block(norm1(x), ...)
            #   x = x + _ff_block(norm2(x), ...)
            normed = layer.norm1(hidden)
            attn_out, attn_weights = layer.self_attn(
                normed, normed, normed,
                need_weights=True,
                average_attn_weights=False,
            )
            attention_maps.append(attn_weights.detach())
            hidden = hidden + layer.dropout1(attn_out)
            # FFN block
            hidden = hidden + layer._ff_block(layer.norm2(hidden))

        # Final norm
        hidden = model.encoder.norm(hidden)

    return attention_maps


def attention_to_channel_importance(
    attention_map: Tensor,
    n_channels: int,
    n_patches: int,
) -> NDArray[np.float64]:
    """Convert attention map to per-channel importance scores.

    Averages attention across heads and batch, then maps token indices
    back to channels (token_i → channel = i // n_patches).

    Parameters
    ----------
    attention_map : Tensor (B, H, N, N) or (B, N, N)
    n_channels : int
    n_patches : int

    Returns
    -------
    channel_importance : array (n_channels,) — normalised importance.
    """
    attn = attention_map.detach().cpu().float().numpy()

    # Average over batch and heads
    while attn.ndim > 2:
        attn = attn.mean(axis=0)

    # attn is now (N, N) — average across query dimension
    token_importance = attn.mean(axis=0)  # (N,)

    # Map to channels
    channel_importance = np.zeros(n_channels)
    for tok_idx, imp in enumerate(token_importance):
        ch = tok_idx // n_patches
        if ch < n_channels:
            channel_importance[ch] += imp

    # Normalise
    total = channel_importance.sum()
    if total > 0:
        channel_importance /= total

    return channel_importance


# ── Visualization (matplotlib) ───────────────────────────────────────

def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> Any:
    """Plot a confusion matrix.

    Returns the matplotlib figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = ["Left", "Right", "Feet", "Tongue"]

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=14)
    fig.colorbar(im)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Annotate cells
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j, i, str(cm_arr[i, j]),
                ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_channel_attention(
    channel_importance: NDArray[np.float64],
    channel_names: List[str],
    title: str = "Channel Attention Map",
    save_path: Optional[str] = None,
) -> Any:
    """Bar chart of per-channel attention importance.

    Highlights C3/C4 if present (motor imagery relevant).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    n = len(channel_names)
    colors = []
    for name in channel_names:
        if name in ("C3", "C4"):
            colors.append("#e74c3c")  # highlight
        else:
            colors.append("#3498db")

    ax.bar(range(n), channel_importance, color=colors, edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(channel_names, rotation=45, fontsize=8)
    ax.set_ylabel("Attention Importance")
    ax.set_title(title, fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="C3/C4 (motor)"),
        Patch(facecolor="#3498db", label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_loss_curve(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
) -> Any:
    """Plot training loss curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=2, color="#2c3e50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
