from pathlib import Path

import numpy as np

from koopman_jepa.paper_config import (
    PaperOptimizationConfig,
    load_paper_experiment_config,
)


def test_overfit_smoke_config_is_small_balanced_and_explicit() -> None:
    path = Path(__file__).parents[1] / "configs" / "paper_overfit_smoke.yaml"

    config = load_paper_experiment_config(path)

    assert config.data.train_per_regime == 1
    assert config.data.normalization == "per_sequence"
    assert config.model.encoder_projection == "direct"
    assert config.model.predictor_kind == "linear"
    assert config.model.linear_initialization == "identity"
    assert config.optimization.optimizer == "adamw"
    assert config.optimization.batch_size == 18
    assert config.optimization.steps == 100
    assert config.optimization.device == "cpu"
    assert config.gate.comparison_window == 10
    assert config.gate.max_loss_ratio == 0.25
    assert config.gate.min_embedding_std_ratio == 0.10
    assert config.gate.min_effective_rank == 2.0


def test_optimization_config_rejects_invalid_values() -> None:
    invalid = PaperOptimizationConfig(steps=0)

    with np.testing.assert_raises_regex(ValueError, "steps must be positive"):
        invalid.validate()
