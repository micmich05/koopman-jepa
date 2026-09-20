from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

EncoderProjection = Literal["direct", "two_stage"]
PredictorKind = Literal["linear", "mlp"]
LinearInitialization = Literal["identity", "xavier_uniform"]
MLPDepth = Literal["one_hidden", "two_hidden"]


@dataclass(frozen=True, slots=True)
class PaperModelConfig:
    latent_dim: int = 32
    input_length: int = 768
    encoder_projection: EncoderProjection = "direct"
    predictor_kind: PredictorKind = "linear"
    linear_initialization: LinearInitialization = "identity"
    mlp_depth: MLPDepth = "two_hidden"
    ema_decay: float = 0.996

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.input_length != 768:
            raise ValueError("the published encoder expects input_length=768")
        if self.encoder_projection not in {"direct", "two_stage"}:
            raise ValueError(f"unknown encoder projection: {self.encoder_projection}")
        if self.predictor_kind not in {"linear", "mlp"}:
            raise ValueError(f"unknown predictor kind: {self.predictor_kind}")
        if self.linear_initialization not in {"identity", "xavier_uniform"}:
            raise ValueError(
                f"unknown linear initialization: {self.linear_initialization}"
            )
        if self.mlp_depth not in {"one_hidden", "two_hidden"}:
            raise ValueError(f"unknown MLP depth: {self.mlp_depth}")
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError("ema_decay must be between zero and one")


class PaperTemporalEncoder(nn.Module):
    """Published four-block convolutional encoder with named projection variants."""

    flattened_dim = 128 * 48

    def __init__(
        self,
        latent_dim: int = 32,
        projection: EncoderProjection = "direct",
        input_length: int = 768,
    ) -> None:
        super().__init__()
        if input_length != 768:
            raise ValueError("the published encoder expects input_length=768")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if projection not in {"direct", "two_stage"}:
            raise ValueError(f"unknown encoder projection: {projection}")

        self.input_length = input_length
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        if projection == "direct":
            self.projection = nn.Linear(self.flattened_dim, latent_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(self.flattened_dim, 2 * latent_dim),
                nn.ReLU(),
                nn.Linear(2 * latent_dim, latent_dim),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected_shape = (1, self.input_length)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected inputs shaped (batch, 1, {self.input_length}); "
                f"received {tuple(inputs.shape)}"
            )
        return self.projection(self.features(inputs))


class PaperLinearPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        initialization: LinearInitialization = "identity",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim, bias=False)
        if initialization == "identity":
            nn.init.eye_(self.linear.weight)
        elif initialization == "xavier_uniform":
            nn.init.xavier_uniform_(self.linear.weight)
        else:
            raise ValueError(f"unknown linear initialization: {initialization}")

    @property
    def matrix(self) -> torch.Tensor:
        return self.linear.weight

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


class PaperMLPPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        depth: MLPDepth = "two_hidden",
    ) -> None:
        super().__init__()
        hidden_dim = 2 * latent_dim
        if depth == "one_hidden":
            layers: list[nn.Module] = [
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            ]
        elif depth == "two_hidden":
            layers = [
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            ]
        else:
            raise ValueError(f"unknown MLP depth: {depth}")
        self.layers = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings)


class PaperTemporalJEPA(nn.Module):
    def __init__(self, config: PaperModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.online_encoder = PaperTemporalEncoder(
            latent_dim=config.latent_dim,
            projection=config.encoder_projection,
            input_length=config.input_length,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

        if config.predictor_kind == "linear":
            self.predictor: nn.Module = PaperLinearPredictor(
                latent_dim=config.latent_dim,
                initialization=config.linear_initialization,
            )
        else:
            self.predictor = PaperMLPPredictor(
                latent_dim=config.latent_dim,
                depth=config.mlp_depth,
            )

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        online_embedding = self.online_encoder(context)
        prediction = self.predictor(online_embedding)
        with torch.no_grad():
            target_embedding = self.target_encoder(target)
        return online_embedding, prediction, target_embedding

    @torch.no_grad()
    def update_target(self) -> None:
        decay = self.config.ema_decay
        for online_parameter, target_parameter in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)
