import math

import numpy as np
import torch
from torch.utils.data import TensorDataset

from koopman_jepa.paper_config import (
    PaperOptimizationConfig,
    PaperOverfitGateConfig,
    PaperScaleInvariantStabilityGateConfig,
    PaperSeedStabilityGateConfig,
    PaperSeedSweepConfig,
    PaperTrainConfig,
    PaperValidationGateConfig,
)
from koopman_jepa.paper_model import PaperModelConfig, PaperTemporalJEPA
from koopman_jepa.paper_training import (
    PaperEpochMetrics,
    PaperSeedSummary,
    PaperStepMetrics,
    evaluate_overfit_gate,
    evaluate_scale_invariant_seed_stability_gate,
    evaluate_seed_stability_gate,
    evaluate_validation_gate,
    make_paper_optimizer,
    overfit_fixed_batch,
    paper_train_step,
    paper_trainable_parameters,
    run_paper_train_validation,
    select_validation_checkpoint,
    squared_embedding_error,
    summarize_seed_checkpoint,
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


def test_checkpoint_selection_rejects_lower_loss_with_excessive_gap() -> None:
    initial = PaperEpochMetrics(
        epoch=0,
        train_loss=0.0067,
        validation_loss=0.0068,
        train_embedding_std=0.009,
        validation_embedding_std=0.009,
        train_effective_rank=22.0,
        validation_effective_rank=21.0,
        online_gradient_norm=0.0,
        predictor_gradient_norm=0.0,
    )
    eligible = PaperEpochMetrics(
        epoch=9,
        train_loss=0.0005,
        validation_loss=0.0017,
        train_embedding_std=0.006,
        validation_embedding_std=0.006,
        train_effective_rank=21.0,
        validation_effective_rank=20.0,
        online_gradient_norm=0.02,
        predictor_gradient_norm=0.002,
    )
    excessive_gap = PaperEpochMetrics(
        epoch=10,
        train_loss=0.00037,
        validation_loss=0.00165,
        train_embedding_std=0.006,
        validation_embedding_std=0.006,
        train_effective_rank=22.0,
        validation_effective_rank=21.0,
        online_gradient_norm=0.02,
        predictor_gradient_norm=0.002,
    )

    selection = select_validation_checkpoint(
        [initial, eligible, excessive_gap],
        PaperValidationGateConfig(),
    )

    assert selection is not None
    assert selection.epoch == 9
    assert selection.validation_loss == eligible.validation_loss
    assert selection.validation_train_loss_ratio == 3.4


def test_seed_stability_gate_requires_every_seed_and_low_variability() -> None:
    sweep = PaperSeedSweepConfig()
    gate = PaperSeedStabilityGateConfig()
    summaries = [
        PaperSeedSummary(
            seed=seed,
            checkpoint_epoch=8 + seed % 2,
            train_loss=0.0005,
            validation_loss=0.0016 + seed * 0.00002,
            validation_loss_ratio=0.24 + seed * 0.005,
            validation_train_loss_ratio=3.2 + seed * 0.1,
            validation_embedding_std_ratio=0.65,
            validation_effective_rank=19.0,
        )
        for seed in sweep.seeds
    ]

    result = evaluate_seed_stability_gate(summaries, sweep, gate)
    missing = evaluate_seed_stability_gate(summaries[:-1], sweep, gate)

    assert result.passed
    assert result.all_checkpoints_selected
    assert result.validation_loss_coefficient_of_variation < 0.25
    assert not missing.passed
    assert not missing.all_checkpoints_selected


def test_scale_invariant_gate_uses_relative_not_absolute_loss_variability() -> None:
    sweep = PaperSeedSweepConfig(seeds=(5, 6, 7, 8, 9))
    gate = PaperScaleInvariantStabilityGateConfig()
    absolute_losses = (0.001, 0.002, 0.003, 0.004, 0.005)
    relative_losses = (0.20, 0.21, 0.22, 0.21, 0.20)
    summaries = [
        PaperSeedSummary(
            seed=seed,
            checkpoint_epoch=9,
            train_loss=validation_loss / 3.0,
            validation_loss=validation_loss,
            validation_loss_ratio=validation_loss_ratio,
            validation_train_loss_ratio=3.0,
            validation_embedding_std_ratio=0.65,
            validation_effective_rank=19.0,
        )
        for seed, validation_loss, validation_loss_ratio in zip(
            sweep.seeds,
            absolute_losses,
            relative_losses,
            strict=True,
        )
    ]

    result = evaluate_scale_invariant_seed_stability_gate(summaries, sweep, gate)
    missing = evaluate_scale_invariant_seed_stability_gate(
        summaries[:-1],
        sweep,
        gate,
    )

    assert result.passed
    assert result.variability_passed
    assert result.validation_loss_ratio_coefficient_of_variation < 0.25
    assert result.absolute_validation_loss_coefficient_of_variation > 0.25
    assert not missing.passed
    assert not missing.all_checkpoints_selected


def test_scale_invariant_gate_rejects_variable_relative_improvement() -> None:
    sweep = PaperSeedSweepConfig(seeds=(5, 6, 7, 8, 9))
    gate = PaperScaleInvariantStabilityGateConfig()
    relative_losses = (0.05, 0.10, 0.20, 0.30, 0.40)
    summaries = [
        PaperSeedSummary(
            seed=seed,
            checkpoint_epoch=9,
            train_loss=0.0005,
            validation_loss=0.001,
            validation_loss_ratio=validation_loss_ratio,
            validation_train_loss_ratio=2.0,
            validation_embedding_std_ratio=0.65,
            validation_effective_rank=19.0,
        )
        for seed, validation_loss_ratio in zip(
            sweep.seeds,
            relative_losses,
            strict=True,
        )
    ]

    result = evaluate_scale_invariant_seed_stability_gate(summaries, sweep, gate)

    assert not result.passed
    assert not result.variability_passed
    assert result.validation_loss_ratio_coefficient_of_variation > 0.25


def test_seed_summary_preserves_constraint_eligible_selection() -> None:
    history = [
        PaperEpochMetrics(
            epoch=0,
            train_loss=1.0,
            validation_loss=1.0,
            train_embedding_std=0.5,
            validation_embedding_std=0.5,
            train_effective_rank=8.0,
            validation_effective_rank=8.0,
            online_gradient_norm=0.0,
            predictor_gradient_norm=0.0,
        ),
        PaperEpochMetrics(
            epoch=1,
            train_loss=0.1,
            validation_loss=0.2,
            train_embedding_std=0.4,
            validation_embedding_std=0.4,
            train_effective_rank=6.0,
            validation_effective_rank=6.0,
            online_gradient_norm=1.0,
            predictor_gradient_norm=1.0,
        ),
    ]
    selection = select_validation_checkpoint(history, PaperValidationGateConfig())

    assert selection is not None
    summary = summarize_seed_checkpoint(3, selection)

    assert summary.seed == 3
    assert summary.checkpoint_epoch == 1
    assert summary.validation_loss_ratio == 0.2
