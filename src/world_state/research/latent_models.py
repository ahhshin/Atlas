from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def padded_shape(height: int, width: int, patch_size: int) -> tuple[int, int]:
    return (
        ((height + patch_size - 1) // patch_size) * patch_size,
        ((width + patch_size - 1) // patch_size) * patch_size,
    )


class RegionalEncoder(nn.Module):
    """Encode physical fields into one vector per geographic patch."""

    def __init__(
        self,
        physical_channels: int,
        latent_dimensions: int,
        patch_size: int,
        hidden_channels: int = 48,
    ) -> None:
        super().__init__()
        self.physical_channels = physical_channels
        self.latent_dimensions = latent_dimensions
        self.patch_size = patch_size
        self.network = nn.Sequential(
            nn.Conv2d(physical_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                latent_dimensions,
                kernel_size=patch_size,
                stride=patch_size,
            ),
        )

    def forward(self, values: Tensor, missing_mask: Tensor) -> Tensor:
        if values.shape != missing_mask.shape:
            raise ValueError("physical values and missing mask must have identical shapes")
        if values.ndim != 4 or values.shape[1] != self.physical_channels:
            raise ValueError("encoder input must be [batch, channel, latitude, longitude]")
        height, width = values.shape[-2:]
        padded_height, padded_width = padded_shape(height, width, self.patch_size)
        padding = (0, padded_width - width, 0, padded_height - height)
        combined = torch.cat((values, missing_mask.to(values.dtype)), dim=1)
        if any(padding):
            combined = F.pad(combined, padding)
        return self.network(combined)

    def encode_sequence(self, values: Tensor, missing_mask: Tensor) -> Tensor:
        if values.ndim != 5:
            raise ValueError("sequence input must be [batch, time, channel, latitude, longitude]")
        batch, steps, channels, height, width = values.shape
        flattened = values.reshape(batch * steps, channels, height, width)
        flattened_mask = missing_mask.reshape(batch * steps, channels, height, width)
        latent = self(flattened, flattened_mask)
        return latent.reshape(batch, steps, *latent.shape[1:])


class RegionalDecoder(nn.Module):
    """Decode regional vectors back to the padded physical grid, then crop exactly."""

    def __init__(
        self,
        physical_channels: int,
        latent_dimensions: int,
        patch_size: int,
        hidden_channels: int = 48,
    ) -> None:
        super().__init__()
        self.physical_channels = physical_channels
        self.patch_size = patch_size
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dimensions,
                hidden_channels,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, physical_channels, kernel_size=3, padding=1),
        )

    def forward(self, latent: Tensor, output_shape: tuple[int, int]) -> Tensor:
        decoded = self.network(latent)
        height, width = output_shape
        if decoded.shape[-2] < height or decoded.shape[-1] < width:
            raise ValueError("decoder output is smaller than requested physical shape")
        return decoded[..., :height, :width]


class ConvGRUCell(nn.Module):
    def __init__(self, dimensions: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dimensions = dimensions
        self.gates = nn.Conv2d(
            dimensions * 2, dimensions * 2, kernel_size=kernel_size, padding=padding
        )
        self.candidate = nn.Conv2d(
            dimensions * 2, dimensions, kernel_size=kernel_size, padding=padding
        )

    def forward(self, values: Tensor, hidden: Tensor) -> Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat((values, hidden), dim=1))).chunk(
            2, dim=1
        )
        candidate = torch.tanh(self.candidate(torch.cat((values, reset * hidden), dim=1)))
        return (1 - update) * hidden + update * candidate


class LatentDynamics(nn.Module):
    """Compact temporal recurrence with 3x3 regional interactions."""

    def __init__(self, latent_dimensions: int) -> None:
        super().__init__()
        self.cell = ConvGRUCell(latent_dimensions)
        self.head = nn.Sequential(
            nn.Conv2d(latent_dimensions, latent_dimensions, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_dimensions, latent_dimensions, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 5:
            raise ValueError("latent history must be [batch, time, channel, latitude, longitude]")
        hidden = torch.zeros_like(history[:, 0])
        for step in range(history.shape[1]):
            hidden = self.cell(history[:, step], hidden)
        return history[:, -1] + self.head(hidden)


class FrozenLatentProbe(nn.Module):
    def __init__(self, latent_dimensions: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(latent_dimensions, 1, kernel_size=1)

    def forward(self, latent: Tensor, output_shape: tuple[int, int]) -> Tensor:
        logits = self.head(latent)
        return F.interpolate(logits, size=output_shape, mode="bilinear", align_corners=False)[:, 0]


def masked_loss(
    prediction: Tensor,
    target: Tensor,
    missing_mask: Tensor,
    *,
    kind: str = "huber",
) -> Tensor:
    if prediction.shape != target.shape or target.shape != missing_mask.shape:
        raise ValueError("prediction, target, and missing mask must share shape")
    if kind == "mse":
        values = (prediction - target).square()
    elif kind == "huber":
        values = F.smooth_l1_loss(prediction, target, reduction="none")
    else:
        raise ValueError(f"unknown reconstruction loss: {kind}")
    valid = (~missing_mask.bool()).to(values.dtype)
    denominator = valid.sum().clamp_min(1)
    return (values * valid).sum() / denominator


def parameter_count(*modules: nn.Module) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def trainable_parameter_count(*modules: nn.Module) -> int:
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
