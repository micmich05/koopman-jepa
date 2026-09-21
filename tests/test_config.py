from pathlib import Path

import pytest

from koopman_jepa.config import ExperimentConfig, TrainConfig, load_config, validate_config


def test_smoke_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "phase0_smoke.yaml"
    config = load_config(path)
    validate_config(config)

    assert config.data.context_length == 128
    assert config.model.predictor_init == "identity"
    assert config.train.seed == 0


def test_predictor_learning_rate_multiplier_must_be_positive() -> None:
    config = ExperimentConfig(
        train=TrainConfig(predictor_learning_rate_multiplier=0.0)
    )

    with pytest.raises(
        ValueError,
        match="predictor_learning_rate_multiplier must be positive",
    ):
        validate_config(config)
