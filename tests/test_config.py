from pathlib import Path

from koopman_jepa.config import load_config, validate_config


def test_smoke_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "phase0_smoke.yaml"
    config = load_config(path)
    validate_config(config)

    assert config.data.context_length == 128
    assert config.model.predictor_init == "identity"
    assert config.train.seed == 0
