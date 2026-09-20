import math

import numpy as np
import torch
from torch.utils.data import TensorDataset

from koopman_jepa.paper_config import (
    PaperOptimizationConfig,
    PaperOverfitGateConfig,
    PaperTrainConfig,
    PaperValidationGateConfig,
)
from koopman_jepa.paper_model import PaperModelConfig, PaperTemporalJEPA
from koopman_jepa.paper_training import (
    PaperEpochMetrics,
    PaperStepMetrics,
    evaluate_overfit_gate,
    evaluate_validation_gate,
    make_paper_optimizer,
    overfit_fixed_batch,
    paper_train_step,
    paper_trainable_parameters,
    run_paper_train_validation,
    squared_embedding_error,
)


def test_squared_embedding_error_matches_mean_squared_l2_distance() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    loss = squared_embedding_error(prediction, target)

    assert loss.item() == 9.0


def test_squared_embedding_error_rejects_incompatible_embeddings() -> None:
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        squared_embedding_error(torch.zeros(2, 3), torch.zeros(2, 4))


def test_trainable_parameters_exclude_ema_target() -> None:
    model = PaperTemporalJEPA(PaperModelConfig(latent_dim=4))
    trainable_ids = {id(parameter) for parameter in paper_trainable_parameters(model)}
    online_ids = {id(parameter) for parameter in model.online_encoder.parameters()}
    predictor_ids = {id(parameter) for parameter in model.predictor.parameters()}
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}

    assert trainable_ids == online_ids | predictor_ids
    assert trainable_ids.isdisjoint(target_ids)


def test_one_batch_smoke_updates_online_predictor_then_ema_target() -> None:
    torch.manual_seed(7)
    config = PaperModelConfig(latent_dim=4, ema_decay=0.996)
    model = PaperTemporalJEPA(config)
    optimizer = torch.optim.SGD(paper_trainable_parameters(model), lr=1e-4)
    context = torch.randn(2, 1, 768)
    target = torch.randn(2, 1, 768)

    online_before = [
        parameter.detach().clone() for parameter in model.online_encoder.parameters()
    ]
    predictor_before = [
        parameter.detach().clone() for parameter in model.predictor.parameters()
    ]
    target_before = [
        parameter.detach().clone() for parameter in model.target_encoder.parameters()
    ]

    metrics = paper_train_step(model, context, target, optimizer)

    assert math.isfinite(metrics.loss)
    assert metrics.loss > 0.0
    assert metrics.online_gradient_norm > 0.0
    assert metrics.predictor_gradient_norm > 0.0
    assert metrics.embedding_std_mean > 0.0
    assert 1.0 <= metrics.effective_rank <= 2.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            online_before,
            model.online_encoder.parameters(),
            strict=True,
        )
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            predictor_before,
            model.predictor.parameters(),
            strict=True,
        )
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    for before, online, ema_target in zip(
        target_before,
        model.online_encoder.parameters(),
        model.target_encoder.parameters(),
        strict=True,
    ):
        expected = config.ema_decay * before + (1.0 - config.ema_decay) * online
        assert torch.allclose(ema_target, expected, atol=1e-7)


def test_fixed_batch_runner_uses_frozen_optimizer_and_step_count() -> None:
    torch.manual_seed(9)
    model = PaperTemporalJEPA(PaperModelConfig(latent_dim=4))
    config = PaperOptimizationConfig(batch_size=2, steps=2)
    optimizer = make_paper_optimizer(model, config)
    context = torch.randn(2, 1, 768)
    target = torch.randn(2, 1, 768)

    history = overfit_fixed_batch(model, context, target, optimizer, config)

    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_ids == {id(parameter) for parameter in paper_trainable_parameters(model)}
    assert len(history) == 2
    assert all(math.isfinite(row.loss) for row in history)


def test_overfit_gate_requires_loss_reduction_without_collapse() -> None:
    healthy_history = [
        PaperStepMetrics(
            loss=1.0 if step < 10 else 0.1,
            online_gradient_norm=1.0,
            predictor_gradient_norm=1.0,
            embedding_std_mean=0.5,
            effective_rank=3.0,
        )
        for step in range(20)
    ]
    collapsed_history = [
        PaperStepMetrics(
            loss=1.0 if step < 10 else 0.1,
            online_gradient_norm=1.0,
            predictor_gradient_norm=1.0,
            embedding_std_mean=0.5 if step < 10 else 0.01,
            effective_rank=1.0 if step >= 10 else 3.0,
        )
        for step in range(20)
    ]
    gate = PaperOverfitGateConfig(comparison_window=10)

    healthy = evaluate_overfit_gate(healthy_history, gate)
    collapsed = evaluate_overfit_gate(collapsed_history, gate)

    assert healthy.passed
    assert healthy.loss_ratio == 0.1
    assert not collapsed.passed
    assert collapsed.loss_passed
    assert not collapsed.spread_passed
    assert not collapsed.rank_passed


def test_train_validation_runner_records_untrained_and_trained_epochs() -> None:
    torch.manual_seed(13)
    train_dataset = TensorDataset(
        torch.randn(18, 1, 768),
        torch.randn(18, 1, 768),
        torch.arange(18),
    )
    validation_dataset = TensorDataset(
        torch.randn(12, 1, 768),
        torch.randn(12, 1, 768),
        torch.arange(12),
    )
    model = PaperTemporalJEPA(PaperModelConfig(latent_dim=4))
    config = PaperTrainConfig(batch_size=6, epochs=1, learning_rate=1e-4)

    history = run_paper_train_validation(
        model,
        train_dataset,
        validation_dataset,
        config,
    )

    assert [row.epoch for row in history] == [0, 1]
    assert history[0].online_gradient_norm == 0.0
    assert history[0].predictor_gradient_norm == 0.0
    assert history[1].online_gradient_norm > 0.0
    assert history[1].predictor_gradient_norm > 0.0
    assert all(math.isfinite(row.train_loss) for row in history)
    assert all(math.isfinite(row.validation_loss) for row in history)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_validation_gate_checks_improvement_gap_and_collapse() -> None:
    initial = PaperEpochMetrics(
        epoch=0,
        train_loss=1.0,
        validation_loss=1.0,
        train_embedding_std=0.5,
        validation_embedding_std=0.5,
        train_effective_rank=8.0,
        validation_effective_rank=8.0,
        online_gradient_norm=0.0,
        predictor_gradient_norm=0.0,
    )
    healthy_final = PaperEpochMetrics(
        epoch=1,
        train_loss=0.1,
        validation_loss=0.2,
        train_embedding_std=0.4,
        validation_embedding_std=0.4,
        train_effective_rank=6.0,
        validation_effective_rank=6.0,
        online_gradient_norm=1.0,
        predictor_gradient_norm=1.0,
    )
    collapsed_final = PaperEpochMetrics(
        epoch=1,
        train_loss=0.1,
        validation_loss=0.2,
        train_embedding_std=0.01,
        validation_embedding_std=0.01,
        train_effective_rank=1.0,
        validation_effective_rank=1.0,
        online_gradient_norm=1.0,
        predictor_gradient_norm=1.0,
    )
    gate = PaperValidationGateConfig(final_window=1)

    healthy = evaluate_validation_gate([initial, healthy_final], gate)
    collapsed = evaluate_validation_gate([initial, collapsed_final], gate)

    assert healthy.passed
    assert healthy.validation_loss_ratio == 0.2
    assert healthy.validation_train_loss_ratio == 2.0
    assert not collapsed.passed
    assert collapsed.loss_passed
    assert collapsed.generalization_gap_passed
    assert not collapsed.spread_passed
    assert not collapsed.rank_passed
