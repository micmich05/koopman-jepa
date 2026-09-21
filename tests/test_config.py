from pathlib import Path

import pytest

from koopman_jepa.config import ExperimentConfig, TrainConfig, load_config, validate_config


def test_minimal_training_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(
        """\
data:
  context_length: 128
model:
  predictor_init: random
train:
  seed: 7
""",
        encoding="utf-8",
    )
    config = load_config(path)
    validate_config(config)

    assert config.data.context_length == 128
    assert config.model.predictor_init == "random"
    assert config.train.seed == 7


def test_predictor_learning_rate_multiplier_must_be_positive() -> None:
    config = ExperimentConfig(
        train=TrainConfig(predictor_learning_rate_multiplier=0.0)
    )

    with pytest.raises(
        ValueError,
        match="predictor_learning_rate_multiplier must be positive",
    ):
        validate_config(config)


@pytest.mark.parametrize("freeze_epoch", [0, 2, 3])
def test_encoder_freeze_epoch_must_be_inside_training(freeze_epoch: int) -> None:
    config = ExperimentConfig(
        train=TrainConfig(epochs=2, freeze_encoder_after_epoch=freeze_epoch)
    )

    with pytest.raises(
        ValueError,
        match="freeze_encoder_after_epoch must be within training",
    ):
        validate_config(config)
