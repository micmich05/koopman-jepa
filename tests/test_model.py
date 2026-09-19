import torch

from koopman_jepa.model import LinearPredictor, TemporalEncoder, TemporalJEPA


def test_identity_predictor_is_exact_at_initialization() -> None:
    predictor = LinearPredictor(latent_dim=4, initialization="identity")
    assert torch.allclose(predictor.matrix, torch.eye(4))


def test_encoder_supports_a_single_convolutional_stage() -> None:
    encoder = TemporalEncoder(latent_dim=3, channels=[4])
    assert encoder(torch.randn(2, 1, 32)).shape == (2, 3)


def test_ema_update_moves_target_toward_online() -> None:
    model = TemporalJEPA(latent_dim=3, channels=[4, 8], predictor_init="identity")
    with torch.no_grad():
        for parameter in model.online_encoder.parameters():
            parameter.add_(1.0)

    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    online = [parameter.clone() for parameter in model.online_encoder.parameters()]
    model.update_target(momentum=0.5)

    for previous, current_online, current_target in zip(
        before,
        online,
        model.target_encoder.parameters(),
        strict=True,
    ):
        expected = 0.5 * previous + 0.5 * current_online
        assert torch.allclose(current_target, expected)


def test_model_shapes() -> None:
    model = TemporalJEPA(latent_dim=5, channels=[4, 8], predictor_init="random")
    context = torch.randn(7, 1, 64)
    target = torch.randn(7, 1, 64)
    online, prediction, target_embedding = model(context, target)

    assert online.shape == (7, 5)
    assert prediction.shape == (7, 5)
    assert target_embedding.shape == (7, 5)
    assert not target_embedding.requires_grad
