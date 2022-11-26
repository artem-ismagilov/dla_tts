import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .variance_predictor import VariancePredictor
from .length_regulator import LengthRegulator


class QuantizationEmbedding(nn.Module):
    def __init__(self, quantization_type, n_bins, min_val, max_val, hidden):
        super().__init__()

        assert quantization_type in ['log', 'linear']
        if quantization_type == 'log':
            self.bounds = torch.exp(torch.linspace(np.log(min_val + 1e-8), np.log(max_val), n_bins - 1)),
        else:
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
            'linear',
            model_config.variance_embedding_bins,
            *model_config.energy_min_max,
            model_config.encoder_dim,
        )

    def get_variance_embedding(self, emb, x, target):
        if target is not None:
            return emb(target)
        else:
            return emb(x)

    def forward(
        self,
        x,
        max_len=None,
        duration_target=None,
        energy_target=None,
        alpha=1.0,
        energy_alpha=1.0):

        x, duration_prediction = self.length_regulator(x, alpha, duration_target, max_len)

        energy_prediction = self.energy_predictor(x) * energy_alpha
        x = x + self.get_variance_embedding(self.energy_emb, x, energy_target)

        return (
            x,
            duration_prediction,
            energy_prediction,
        )
