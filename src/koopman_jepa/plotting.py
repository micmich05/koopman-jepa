from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, train, label="train")
    axis.plot(epochs, val, label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_yscale("log")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    regime_names: tuple[str, ...],
    output_path: Path,
) -> None:
    projection = PCA(n_components=2).fit_transform(embeddings)
    figure, axis = plt.subplots(figsize=(7, 5))
    for regime_id, regime_name in enumerate(regime_names):
        selected = labels == regime_id
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            s=10,
            alpha=0.65,
            label=regime_name,
        )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_predictor_spectrum(
    matrix: np.ndarray,
    output_path: Path,
    active_eigenvalues: np.ndarray | None = None,
) -> None:
    full_eigenvalues = np.linalg.eigvals(matrix)
    angle = np.linspace(0.0, 2.0 * np.pi, 300)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(np.cos(angle), np.sin(angle), linestyle="--", color="0.75", linewidth=1)
    axis.scatter(
        full_eigenvalues.real,
        full_eigenvalues.imag,
        color="0.7",
        s=30,
        label="full latent space",
    )
    if active_eigenvalues is not None and active_eigenvalues.size > 0:
        axis.scatter(
            active_eigenvalues.real,
            active_eigenvalues.imag,
            color="tab:blue",
            s=42,
            label="active centroid span",
        )
    axis.scatter([1.0], [0.0], marker="x", color="tab:red", s=70, label="target λ=1")
    axis.axhline(0.0, color="0.85", linewidth=1)
    axis.axvline(0.0, color="0.85", linewidth=1)
    axis.set_xlabel("real")
    axis.set_ylabel("imaginary")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
