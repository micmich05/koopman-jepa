import numpy as np

from koopman_jepa.plotting import plot_embeddings, plot_history, plot_predictor_spectrum


def test_phase0_plots_are_written(tmp_path) -> None:
    history = [
        {"epoch": 1.0, "train_loss": 1.0, "val_loss": 1.2},
        {"epoch": 2.0, "train_loss": 0.5, "val_loss": 0.7},
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    matrix = np.diag([1.0, 0.9, 0.8])

    plot_history(history, tmp_path / "loss.png")
    plot_embeddings(embeddings, labels, ("a", "b"), tmp_path / "embeddings.png")
    plot_predictor_spectrum(
        matrix,
        tmp_path / "spectrum.png",
        active_eigenvalues=np.array([1.0 + 0.0j, 0.9 + 0.0j]),
    )

    for filename in ("loss.png", "embeddings.png", "spectrum.png"):
        assert (tmp_path / filename).stat().st_size > 0
