from __future__ import annotations

import copy
from typing import Literal

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: list[int],
        pooling: Literal["global", "flatten"] = "global",
        input_length: int | None = None,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not channels:
            raise ValueError("channels cannot be empty")
        if any(channel < 1 for channel in channels):
            raise ValueError("channels must contain only positive values")
        if pooling not in {"global", "flatten"}:
            raise ValueError("pooling must be 'global' or 'flatten'")
        if pooling == "flatten" and (input_length is None or input_length < 1):
            raise ValueError("flatten pooling requires a positive input_length")

        blocks: list[nn.Module] = []
        in_channels = 1
        kernel_sizes = [7, 5, *([3] * max(0, len(channels) - 2))][: len(channels)]
        feature_length = input_length
        for out_channels, kernel_size in zip(channels, kernel_sizes, strict=True):
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
            if feature_length is not None:
                feature_length = (feature_length + 1) // 2

        self.features = nn.Sequential(*blocks)
        self.pooling = pooling
        if pooling == "global":
            self.pool = nn.AdaptiveAvgPool1d(1)
            projection_input = channels[-1]
        else:
            self.pool = nn.Flatten(start_dim=1)
            projection_input = channels[-1] * feature_length  # type: ignore[operator]
        self.projection = nn.Linear(projection_input, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features)
        if self.pooling == "global":
            pooled = pooled.squeeze(-1)
        return self.projection(pooled)


class LinearPredictor(nn.Module):
    def __init__(self, latent_dim: int, initialization: str) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim, bias=False)
        self.reset_parameters(initialization)

    def reset_parameters(self, initialization: str) -> None:
        if initialization == "identity":
            nn.init.eye_(self.linear.weight)
        elif initialization == "random":
            nn.init.xavier_uniform_(self.linear.weight)
        else:
            raise ValueError(f"Unknown predictor initialization: {initialization}")

    @property
    def matrix(self) -> torch.Tensor:
        return self.linear.weight

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


class TemporalJEPA(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: list[int],
        predictor_init: str,
        pooling: Literal["global", "flatten"] = "global",
        input_length: int | None = None,
    ) -> None:
        super().__init__()
        self.online_encoder = TemporalEncoder(
            latent_dim,
            channels,
            pooling=pooling,
            input_length=input_length,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = LinearPredictor(latent_dim, predictor_init)
        self.target_encoder.requires_grad_(False)

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
    def update_target(self, momentum: float) -> None:
        for online_parameter, target_parameter in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)


def variance_loss(embeddings: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(target_std - std).mean()


def mean_loss(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings.mean(dim=0).square().mean()


def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(embeddings.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().sum() / embeddings.shape[1]
