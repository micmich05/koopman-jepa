import json

from koopman_jepa.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from koopman_jepa.phase0 import _gate_checks, run


def test_identity_gate_fails_cleanly_for_a_collapsed_active_subspace() -> None:
    metrics = {
        "effective_rank": 0.0,
        "linear_probe_accuracy": 0.5,
        "active_rank": 0,
        "online_target_centroid_error": 0.0,
        "active_identity_error": None,
    }

    checks = _gate_checks(metrics, predictor_init="identity")

    assert checks["identity_on_active_subspace"] is False


def test_phase0_smoke_writes_complete_artifact_bundle(tmp_path) -> None:
    config = ExperimentConfig(
        data=DataConfig(
            context_length=32,
            shift=8,
            train_per_regime=2,
            val_per_regime=1,
            test_per_regime=1,
        ),
        model=ModelConfig(latent_dim=3, channels=[4], predictor_init="identity"),
        train=TrainConfig(epochs=1, batch_size=6, learning_rate=1e-3, device="cpu"),
        output_dir=str(tmp_path),
    )

    run_dir = run(config, run_name="smoke")

    expected_files = {
        "checkpoint.pt",
        "config.json",
        "embeddings.npz",
        "embeddings_pca.png",
        "history.json",
        "loss.png",
        "metrics.json",
        "predictor_spectrum.png",
    }
    assert {path.name for path in run_dir.iterdir()} == expected_files

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["predictor_init"] == "identity"
    assert set(metrics["gate_checks"]) == {
        "active_rank_at_least_three",
        "identity_on_active_subspace",
        "noncollapsed",
        "online_target_aligned",
        "regimes_separable",
    }
