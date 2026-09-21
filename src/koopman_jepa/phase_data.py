from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .koopman import PhaseDynamics, balanced_phase_transitions

ObservationSide = Literal["current", "future"]


@dataclass(frozen=True, slots=True)
class PhaseWindowConfig:
    """Emission law shared by all phase-dynamics conditions."""

    num_phases: int = 4
    window_length: int = 128
    repeats_per_transition: int = 64
    noise_std: float = 0.08
    amplitude_low: float = 0.8
    amplitude_high: float = 1.2
    offset_std: float = 0.15
    max_shift: int = 4
    pulse_width: float = 0.055
    secondary_offset: float = 0.1875
    secondary_strength: float = 0.45


@dataclass(frozen=True, slots=True)
class PhaseObservationCondition:
    dynamics: PhaseDynamics
    current_windows: np.ndarray
    future_windows: np.ndarray
    current_phases: np.ndarray
    future_phases: np.ndarray
    current_occurrences: np.ndarray
    future_occurrences: np.ndarray


@dataclass(frozen=True, slots=True)
class SharedPhaseObservationBundle:
    config: PhaseWindowConfig
    seed: int
    templates: np.ndarray
    conditions: dict[PhaseDynamics, PhaseObservationCondition]


def validate_phase_window_config(config: PhaseWindowConfig) -> None:
    if config.num_phases < 2:
        raise ValueError("num_phases must be at least two")
    if config.window_length < 16:
        raise ValueError("window_length must be at least 16")
    if config.window_length % config.num_phases:
        raise ValueError("window_length must be divisible by num_phases")
    if config.repeats_per_transition < 1:
        raise ValueError("repeats_per_transition must be positive")
    if config.noise_std < 0.0 or config.offset_std < 0.0:
        raise ValueError("noise and offset scales must be non-negative")
    if not 0.0 < config.amplitude_low <= config.amplitude_high:
        raise ValueError("amplitude range must be positive and ordered")
    if not 0 <= config.max_shift < config.window_length // (2 * config.num_phases):
        raise ValueError("max_shift must be smaller than half a phase spacing")
    if not 0.0 < config.pulse_width < 0.25:
        raise ValueError("pulse_width must lie between zero and 0.25")
    if not 0.0 < config.secondary_offset < 0.5:
        raise ValueError("secondary_offset must lie between zero and 0.5")
    if not 0.0 <= config.secondary_strength <= 1.0:
        raise ValueError("secondary_strength must lie in [0, 1]")


def _circular_distance(coordinate: np.ndarray, center: float) -> np.ndarray:
    return np.mod(coordinate - center + 0.5, 1.0) - 0.5


def phase_window_templates(config: PhaseWindowConfig) -> np.ndarray:
    """Create standardized phase templates related by exact circular shifts."""

    validate_phase_window_config(config)
    coordinate = np.arange(config.window_length, dtype=np.float64) / config.window_length
    templates = []
    for phase in range(config.num_phases):
        center = (phase + 0.5) / config.num_phases
        primary_distance = _circular_distance(coordinate, center)
        secondary_distance = _circular_distance(
            coordinate,
            (center + config.secondary_offset) % 1.0,
        )
        primary = np.exp(-0.5 * (primary_distance / config.pulse_width) ** 2)
        secondary = np.exp(-0.5 * (secondary_distance / config.pulse_width) ** 2)
        template = primary - config.secondary_strength * secondary
        template = (template - template.mean()) / template.std()
        templates.append(template)
    return np.asarray(templates, dtype=np.float32)


def _sample_window_bank(
    templates: np.ndarray,
    samples_per_phase: int,
    config: PhaseWindowConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    shape = (config.num_phases, samples_per_phase, config.window_length)
    amplitudes = rng.uniform(
        config.amplitude_low,
        config.amplitude_high,
        size=shape[:2],
    )
    offsets = rng.normal(0.0, config.offset_std, size=shape[:2])
    shifts = rng.integers(-config.max_shift, config.max_shift + 1, size=shape[:2])
    noise = rng.normal(0.0, config.noise_std, size=shape)

    windows = np.empty(shape, dtype=np.float32)
    for phase in range(config.num_phases):
        for occurrence in range(samples_per_phase):
            shifted = np.roll(templates[phase], shifts[phase, occurrence])
            windows[phase, occurrence] = (
                amplitudes[phase, occurrence] * shifted
                + offsets[phase, occurrence]
                + noise[phase, occurrence]
            )
    return windows


def _occurrence_indices(phases: np.ndarray, num_phases: int) -> np.ndarray:
    counters = np.zeros(num_phases, dtype=np.int64)
    occurrences = np.empty_like(phases)
    for index, phase in enumerate(phases):
        occurrences[index] = counters[phase]
        counters[phase] += 1
    return occurrences


def make_shared_phase_observation_bundle(
    config: PhaseWindowConfig,
    seed: int,
) -> SharedPhaseObservationBundle:
    """Pair identical observation marginals under three different dynamics."""

    validate_phase_window_config(config)
    templates = phase_window_templates(config)
    samples_per_phase = config.num_phases * config.repeats_per_transition
    source_seed, future_seed = np.random.SeedSequence(seed).spawn(2)
    source_bank = _sample_window_bank(
        templates,
        samples_per_phase,
        config,
        np.random.default_rng(source_seed),
    )
    future_bank = _sample_window_bank(
        templates,
        samples_per_phase,
        config,
        np.random.default_rng(future_seed),
    )

    conditions: dict[PhaseDynamics, PhaseObservationCondition] = {}
    for dynamics in ("static", "cyclic", "independent"):
        current_phases, future_phases = balanced_phase_transitions(
            dynamics,
            repeats_per_transition=config.repeats_per_transition,
            num_phases=config.num_phases,
        )
        current_occurrences = _occurrence_indices(current_phases, config.num_phases)
        future_occurrences = _occurrence_indices(future_phases, config.num_phases)
        current_windows = source_bank[current_phases, current_occurrences, None, :]
        future_windows = future_bank[future_phases, future_occurrences, None, :]
        conditions[dynamics] = PhaseObservationCondition(
            dynamics=dynamics,
            current_windows=current_windows,
            future_windows=future_windows,
            current_phases=current_phases,
            future_phases=future_phases,
            current_occurrences=current_occurrences,
            future_occurrences=future_occurrences,
        )

    return SharedPhaseObservationBundle(
        config=config,
        seed=seed,
        templates=templates,
        conditions=conditions,
    )


def ordered_observation_marginal(
    condition: PhaseObservationCondition,
    side: ObservationSide,
) -> np.ndarray:
    """Order a condition marginal by phase and shared emission occurrence."""

    if side == "current":
        windows = condition.current_windows
        phases = condition.current_phases
        occurrences = condition.current_occurrences
    elif side == "future":
        windows = condition.future_windows
        phases = condition.future_phases
        occurrences = condition.future_occurrences
    else:
        raise ValueError(f"unknown observation side: {side}")
    order = np.lexsort((occurrences, phases))
    return windows[order]


def nearest_template_phase_predictions(
    windows: np.ndarray,
    templates: np.ndarray,
    max_shift: int,
) -> np.ndarray:
    """Predict phase after removing per-window amplitude and offset."""

    observations = np.asarray(windows, dtype=np.float64)
    if observations.ndim == 3 and observations.shape[1] == 1:
        observations = observations[:, 0, :]
    if observations.ndim != 2:
        raise ValueError("windows must have shape (samples, length) or (samples, 1, length)")
    references = np.asarray(templates, dtype=np.float64)
    if references.ndim != 2 or references.shape[1] != observations.shape[1]:
        raise ValueError("templates and windows must share their final dimension")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")

    observations = observations - observations.mean(axis=1, keepdims=True)
    observations /= np.maximum(np.linalg.norm(observations, axis=1, keepdims=True), 1e-15)
    scores = np.empty((observations.shape[0], references.shape[0]), dtype=np.float64)
    for phase, template in enumerate(references):
        candidates = np.stack(
            [np.roll(template, shift) for shift in range(-max_shift, max_shift + 1)]
        )
        candidates -= candidates.mean(axis=1, keepdims=True)
        candidates /= np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-15)
        scores[:, phase] = (observations @ candidates.T).max(axis=1)
    return scores.argmax(axis=1)
