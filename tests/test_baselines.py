import numpy as np
import pytest

from koopman_jepa.baselines import fit_ridge_operator, select_ridge_operator


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
