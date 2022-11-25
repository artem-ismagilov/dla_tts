import torch
import torch.nn as nn

from .config import FastSpeechConfig
from .length_regulator import LengthRegulator
from .encoder import Encoder
from .decoder import Decoder


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config=FastSpeechConfig):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        enc_output, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            out, mel_pos = self.length_regulator(x, length_target, mel_max_length)
            out = self.decoder(out, mel_pos)
            out = self.mask_tensor(out, mel_pos, mel_max_length)

            return out, mel_pos
        else:
            out, mel_pos = self.length_regulator(x, alpha)
            out = self.decoder(out, mel_pos)
            return out
