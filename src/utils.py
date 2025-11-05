"""Utility functions."""

import os
import numpy as np
from pathlib import Path
import logging


def create_output_structure(base_dir: str, model_subdir: str, num_layers: int) -> str:
    """
    Create output directory structure.

    Args:
        base_dir: Base output directory
        model_subdir: Model-specific subdirectory
        num_layers: Number of layers

    Returns:
        Full output directory path
    """
    output_dir = Path(base_dir) / model_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating directory structure in {output_dir}...")

    for layer_idx in range(num_layers):
        layer_dir = output_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(exist_ok=True)

    print(f"Created {num_layers} layer directories")

    return str(output_dir)


def aggregate_activations(
    activations: np.ndarray,
    attention_mask: np.ndarray,
    method: str = "mean",
    use_attention_mask: bool = True,
) -> np.ndarray:
    """
    Aggregate activations across sequence dimension.

    Args:
        activations: Shape [batch_size, seq_len, hidden_dim]
        attention_mask: Shape [batch_size, seq_len]
        method: Aggregation method (mean, sum, last, none)
        use_attention_mask: Whether to use attention mask

    Returns:
        Aggregated activations
    """
    if method == "none":
        return activations

    if use_attention_mask:
        # Expand mask to match hidden dimension
        mask = attention_mask[:, :, np.newaxis]  # [batch, seq, 1]

        if method == "mean":
            # Mean pooling with masking
            masked_activations = activations * mask
            sum_activations = np.sum(masked_activations, axis=1)
            count = np.sum(mask, axis=1) + 1e-9
            result = sum_activations / count
        elif method == "sum":
            # Sum pooling with masking
            masked_activations = activations * mask
            result = np.sum(masked_activations, axis=1)
        elif method == "last":
            # Get last non-padded token
            seq_lengths = (attention_mask != 0).cumsum(axis=1).argmax(axis=1)
            result = activations[np.arange(activations.shape[0]), seq_lengths]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    else:
        if method == "mean":
            result = np.mean(activations, axis=1)
        elif method == "sum":
            result = np.sum(activations, axis=1)
        elif method == "last":
            result = activations[:, -1, :]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return result


def get_storage_stats(output_dir: str, num_layers: int) -> dict:
    """
    Calculate storage statistics.

    Args:
        output_dir: Output directory path
        num_layers: Number of layers

    Returns:
        Dictionary with statistics
    """
    output_path = Path(output_dir)

    # Calculate total size
    total_size = 0
    for layer_idx in range(num_layers):
        layer_dir = output_path / f"layer_{layer_idx:02d}"
        if layer_dir.exists():
            layer_size = sum(f.stat().st_size for f in layer_dir.glob("*.npy"))
            total_size += layer_size

    # Load example file
    example_file = output_path / "layer_00" / "batch_0000.npy"
    example_data = np.load(example_file) if example_file.exists() else None

    return {
        "total_size_gb": total_size / (1024**3),
        "avg_per_layer_gb": (
            (total_size / num_layers) / (1024**3) if num_layers > 0 else 0
        ),
        "example_shape": example_data.shape if example_data is not None else None,
        "example_dtype": example_data.dtype if example_data is not None else None,
    }


def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
