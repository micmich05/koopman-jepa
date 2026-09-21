import numpy as np
import pytest
import torch

from koopman_jepa.baselines import (
    collect_encoder_features,
    fit_pca_dmd,
    fit_pca_feature_map,
    fit_random_cnn_dmd,
    fit_raw_window_dmd,
    fit_ridge_operator,
    flatten_windows,
    make_random_cnn_encoder,
    select_ridge_operator,
)
from koopman_jepa.model import TemporalJEPA


def _embedded_linear_windows() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(23)
    dynamics = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    mixing = rng.normal(size=(6, 3))
    offset = np.linspace(-2.0, 3.0, 6)
    train_latent = rng.normal(size=(96, 3))
    train_latent -= train_latent.mean(axis=0, keepdims=True)
    validation_latent = rng.normal(size=(32, 3))

    def observe(latent: np.ndarray) -> np.ndarray:
        return (latent @ mixing.T + offset).reshape(len(latent), 1, -1)

    return (
        observe(train_latent),
        observe(train_latent @ dynamics.T),
        observe(validation_latent),
        observe(validation_latent @ dynamics.T),
    )


def test_fit_ridge_operator_recovers_known_linear_dynamics() -> None:
    rng = np.random.default_rng(7)
    current = rng.normal(size=(64, 3))
    expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )
    future = current @ expected.T

    actual = fit_ridge_operator(current, future, regularization=0.0)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_select_ridge_operator_uses_train_center_and_validation_mse() -> None:
    rng = np.random.default_rng(11)
    expected = np.diag([0.75, -0.5])
    offset = np.array([4.0, -3.0])
    train_current_centered = rng.normal(size=(80, 2))
    train_current_centered -= train_current_centered.mean(axis=0, keepdims=True)
    validation_current_centered = rng.normal(size=(30, 2))
    train_current = train_current_centered + offset
    train_future = train_current_centered @ expected.T + offset
    validation_current = validation_current_centered + offset
    validation_future = validation_current_centered @ expected.T + offset

    fit = select_ridge_operator(
        train_current,
        train_future,
        validation_current,
        validation_future,
        regularizations=[1.0, 0.01, 0.0],
    )

    assert fit.regularization == 0.0
    np.testing.assert_allclose(fit.feature_mean, offset, atol=1e-12)
    np.testing.assert_allclose(fit.matrix, expected, atol=1e-12)
    np.testing.assert_allclose(fit.center(validation_current), validation_current_centered)
    assert fit.validation_mse < 1e-24


def test_select_ridge_operator_breaks_validation_ties_toward_smaller_lambda() -> None:
    zeros = np.zeros((8, 2))

    fit = select_ridge_operator(
        zeros,
        zeros,
        zeros,
        zeros,
        regularizations=[1.0, 0.01, 0.0],
    )

    assert fit.regularization == 0.0
    assert fit.validation_mse_by_regularization == {0.0: 0.0, 0.01: 0.0, 1.0: 0.0}


@pytest.mark.parametrize("regularization", [-1.0, np.inf, np.nan])
def test_fit_ridge_operator_rejects_invalid_regularization(regularization: float) -> None:
    features = np.ones((4, 2))

    with pytest.raises(ValueError, match="finite and non-negative"):
        fit_ridge_operator(features, features, regularization)


def test_select_ridge_operator_rejects_feature_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="feature dimensions must match"):
        select_ridge_operator(
            np.ones((4, 2)),
            np.ones((4, 2)),
            np.ones((4, 3)),
            np.ones((4, 3)),
            regularizations=[0.0],
        )


def test_flatten_windows_preserves_batch_and_flattens_observation_axes() -> None:
    windows = np.arange(24).reshape(3, 2, 4)

    flattened = flatten_windows(windows)

    assert flattened.shape == (3, 8)
    np.testing.assert_array_equal(flattened[1], np.arange(8, 16))


def test_raw_window_dmd_predicts_an_embedded_linear_system() -> None:
    train_current, train_future, validation_current, validation_future = (
        _embedded_linear_windows()
    )

    fit = fit_raw_window_dmd(
        train_current,
        train_future,
        validation_current,
        validation_future,
        regularizations=[0.0, 1e-4, 1e-2],
    )
    current_features = fit.center(flatten_windows(validation_current))
    future_features = fit.center(flatten_windows(validation_future))

    assert fit.regularization == 0.0
    np.testing.assert_allclose(current_features @ fit.matrix.T, future_features, atol=1e-12)


def test_pca3_dmd_predicts_an_embedded_three_dimensional_system() -> None:
    train_current, train_future, validation_current, validation_future = (
        _embedded_linear_windows()
    )

    fit = fit_pca_dmd(
        train_current,
        train_future,
        validation_current,
        validation_future,
        n_components=3,
        regularizations=[0.0, 1e-4, 1e-2],
    )
    current_features = fit.transform(validation_current)
    future_features = fit.transform(validation_future)

    assert fit.operator.regularization == 0.0
    assert fit.feature_map.components.shape == (3, 6)
    np.testing.assert_allclose(
        current_features @ fit.operator.matrix.T,
        future_features,
        atol=1e-12,
    )


def test_pca_feature_map_rejects_too_many_components() -> None:
    windows = np.ones((4, 1, 3))

    with pytest.raises(ValueError, match="n_components"):
        fit_pca_feature_map(windows, windows, n_components=4)


def test_random_encoder_matches_seeded_jepa_online_initialization() -> None:
    seed = 31
    torch.manual_seed(seed)
    jepa = TemporalJEPA(
        latent_dim=3,
        channels=[4, 8],
        predictor_init="random",
        pooling="flatten",
        input_length=32,
    )

    random_encoder = make_random_cnn_encoder(
        seed=seed,
        latent_dim=3,
        channels=[4, 8],
        pooling="flatten",
        input_length=32,
    )

    for random_parameter, jepa_parameter in zip(
        random_encoder.parameters(),
        jepa.online_encoder.parameters(),
        strict=True,
    ):
        assert torch.equal(random_parameter, jepa_parameter)
        assert random_parameter.requires_grad is False


def test_collect_encoder_features_is_batch_size_invariant() -> None:
    encoder = make_random_cnn_encoder(
        seed=37,
        latent_dim=3,
        channels=[4],
        pooling="flatten",
        input_length=16,
    )
    windows = torch.randn(11, 1, 16)

    small_batches = collect_encoder_features(encoder, windows, batch_size=3)
    one_batch = collect_encoder_features(encoder, windows, batch_size=11)

    np.testing.assert_allclose(small_batches, one_batch, atol=1e-7)
    assert small_batches.shape == (11, 3)


def test_random_cnn_dmd_keeps_encoder_frozen() -> None:
    generator = torch.Generator().manual_seed(41)
    train_current = torch.randn(32, 1, 16, generator=generator)
    train_future = torch.roll(train_current, shifts=2, dims=-1)
    validation_current = torch.randn(16, 1, 16, generator=generator)
    validation_future = torch.roll(validation_current, shifts=2, dims=-1)
    reference = make_random_cnn_encoder(
        seed=43,
        latent_dim=3,
        channels=[4],
        pooling="flatten",
        input_length=16,
    )

    fit = fit_random_cnn_dmd(
        train_current,
        train_future,
        validation_current,
        validation_future,
        seed=43,
        latent_dim=3,
        channels=[4],
        pooling="flatten",
        input_length=16,
        batch_size=8,
        regularizations=[0.0, 1e-4, 1e-2],
    )

    for actual, expected in zip(
        fit.encoder.parameters(),
        reference.parameters(),
        strict=True,
    ):
        assert torch.equal(actual, expected)
        assert actual.requires_grad is False
    assert fit.transform(validation_current).shape == (16, 3)
    assert np.isfinite(fit.operator.validation_mse)
