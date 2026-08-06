"""Domain-Adversarial Neural Network (DANN) for era-invariant embeddings."""

import torch
import torch.nn as nn

from .config import (
    DROPOUT,
    ENCODER_HIDDEN,
    ERA_HEAD_HIDDEN,
    LATENT_DIM,
    NUM_ERAS,
    VALUE_HEAD_HIDDEN,
)


class GradientReversalFunction(torch.autograd.Function):
    """Identity in forward; gradient multiplied by -lambda_ in backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class EraTranslatorDANN(nn.Module):
    """Encoder -> [value head] and [GRL -> era head]."""

    def __init__(self, input_dim: int, num_eras: int = NUM_ERAS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ENCODER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(ENCODER_HIDDEN, LATENT_DIM),
            # No final ReLU: a linear latent keeps all dims alive (ReLU dead
            # units collapsed the 16-dim latent to a ~1-dim value axis).
        )
        self.value_head = nn.Sequential(
            nn.Linear(LATENT_DIM, VALUE_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HEAD_HIDDEN, 1),
        )
        self.grl = GradientReversalLayer()
        self.era_head = nn.Sequential(
            nn.Linear(LATENT_DIM, ERA_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(ERA_HEAD_HIDDEN, ERA_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(ERA_HEAD_HIDDEN, num_eras),
        )
        self.num_eras = num_eras
        self.latent_dim = LATENT_DIM

    def forward(self, x, lambda_=1.0):
        z = self.encoder(x)
        value = self.value_head(z).squeeze(-1)
        self.grl.lambda_ = lambda_
        era = self.era_head(self.grl(z))
        return z, value, era
