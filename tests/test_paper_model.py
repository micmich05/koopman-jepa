import numpy as np
import torch
from torch import nn

from koopman_jepa.paper_model import (
    PaperLinearPredictor,
    PaperMLPPredictor,
    PaperModelConfig,
    PaperTemporalEncoder,
    PaperTemporalJEPA,
)


def test_published_convolutional_geometry() -> None:
    encoder = PaperTemporalEncoder()
    convolutions = [layer for layer in encoder.features if isinstance(layer, nn.Conv1d)]
    geometry = [
        (
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size[0],
            layer.stride[0],
            layer.padding[0],
        )
        for layer in convolutions
    ]

    assert geometry == [
        (1, 16, 7, 2, 3),
        (16, 32, 5, 2, 2),
        (32, 64, 3, 2, 1),
        (64, 128, 3, 2, 1),
    ]
    assert encoder.flattened_dim == 6144


def test_encoder_projection_variants_return_published_latent_shape() -> None:
    inputs = torch.randn(3, 1, 768)

    direct = PaperTemporalEncoder(projection="direct")
    two_stage = PaperTemporalEncoder(projection="two_stage")

    assert direct(inputs).shape == (3, 32)
    assert two_stage(inputs).shape == (3, 32)
    assert isinstance(direct.projection, nn.Linear)
    assert direct.projection.in_features == 6144
    assert direct.projection.out_features == 32

    two_stage_linears = [
        layer for layer in two_stage.projection if isinstance(layer, nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in two_stage_linears] == [
        (6144, 64),
        (64, 32),
    ]


def test_encoder_rejects_non_published_input_geometry() -> None:
    encoder = PaperTemporalEncoder()

    with np.testing.assert_raises_regex(ValueError, "expected inputs"):
        encoder(torch.randn(2, 1, 767))


def test_linear_predictor_initialization_variants() -> None:
    identity = PaperLinearPredictor(initialization="identity")
    expected_identity = torch.eye(32)

    torch.manual_seed(4)
    random = PaperLinearPredictor(initialization="xavier_uniform")

    assert identity.linear.bias is None
    assert torch.equal(identity.matrix, expected_identity)
    assert random.linear.bias is None
    assert not torch.equal(random.matrix, expected_identity)


def test_mlp_depth_variants_match_named_architectures() -> None:
    one_hidden = PaperMLPPredictor(depth="one_hidden")
    two_hidden = PaperMLPPredictor(depth="two_hidden")
    inputs = torch.randn(5, 32)

    one_linears = [layer for layer in one_hidden.layers if isinstance(layer, nn.Linear)]
    two_linears = [layer for layer in two_hidden.layers if isinstance(layer, nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in one_linears] == [
        (32, 64),
        (64, 32),
    ]
    assert [(layer.in_features, layer.out_features) for layer in two_linears] == [
        (32, 64),
        (64, 64),
        (64, 32),
    ]
    assert one_hidden(inputs).shape == (5, 32)
    assert two_hidden(inputs).shape == (5, 32)


def test_jepa_forward_shapes_and_frozen_target() -> None:
    model = PaperTemporalJEPA(PaperModelConfig())
    context = torch.randn(2, 1, 768)
    target = torch.randn(2, 1, 768)

    online, prediction, target_embedding = model(context, target)

    assert online.shape == prediction.shape == target_embedding.shape == (2, 32)
    assert online.requires_grad
    assert prediction.requires_grad
    assert not target_embedding.requires_grad
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_target_starts_as_copy_and_uses_published_ema_decay() -> None:
    model = PaperTemporalJEPA(PaperModelConfig(ema_decay=0.996))
    initial_online = [parameter.detach().clone() for parameter in model.online_encoder.parameters()]
    initial_target = [parameter.detach().clone() for parameter in model.target_encoder.parameters()]

    for online, target in zip(initial_online, initial_target, strict=True):
        assert torch.equal(online, target)

    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.add_(1.0)
    model.update_target()

    for before, online, target in zip(
        initial_target,
        model.online_encoder.parameters(),
        model.target_encoder.parameters(),
        strict=True,
    ):
        expected = 0.996 * before + 0.004 * online
        assert torch.allclose(target, expected, atol=1e-7)


def test_model_config_rejects_unknown_variants() -> None:
    config = PaperModelConfig(encoder_projection="unknown")  # type: ignore[arg-type]

    with np.testing.assert_raises_regex(ValueError, "unknown encoder projection"):
        config.validate()
