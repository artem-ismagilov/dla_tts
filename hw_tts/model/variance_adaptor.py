import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .variance_predictor import VariancePredictor
from .length_regulator import LengthRegulator


class QuantizationEmbedding(nn.Module):
    def __init__(self, n_bins, min_val, max_val, hidden):
        super().__init__()

        self.bounds = torch.linspace(min_val, max_val, n_bins - 1)

        self.bounds = nn.Parameter(self.bounds, requires_grad=False)
        self.embedding = nn.Embedding(n_bins, hidden)

    def forward(self, x):
        return self.embedding(torch.bucketize(x, self.bounds))


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.length_regulator = LengthRegulator(model_config)

        self.energy_predictor = VariancePredictor(model_config)
        self.energy_emb = QuantizationEmbedding(
            model_config.variance_embedding_bins,
            *model_config.energy_min_max,
            model_config.encoder_dim,
        )

    def get_variance_embedding(self, emb, energy_prediction, target):
        if target is not None:
            return emb(target)
        else:
            return emb(energy_prediction)

    def forward(
        self,
        x,
        max_len=None,
        duration_target=None,
        energy_target=None,
        alpha=1.0,
        energy_alpha=1.0):

        x, duration_prediction = self.length_regulator(x, alpha, duration_target, max_len)
        duration_prediction = torch.clamp(duration_prediction, min=1e-8)

        energy_prediction = self.energy_predictor(x) * energy_alpha
        x = x + self.get_variance_embedding(self.energy_emb, energy_prediction, energy_target)

        return (
            x,
            duration_prediction,
            energy_prediction,
        )
