import torch
import torch.nn as nn

from .config import FastSpeechConfig
from .variance_adaptor import VarianceAdaptor
from .encoder import Encoder
from .decoder import Decoder
from .util import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config=FastSpeechConfig):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(
        self,
        src_seq,
        src_pos,
        mel_pos=None,
        mel_max_length=None,
        length_target=None,
        energy_target=None,
        alpha=1.0,
        energy_alpha=1.0):

        enc_output, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            out, dur_out, energy_out = self.variance_adaptor(
                enc_output,
                mel_max_length,
                length_target,
                energy_target,
                alpha,
                energy_alpha,
            )

            out = self.decoder(out, mel_pos)
            out = self.mask_tensor(out, mel_pos, mel_max_length)
            out = self.mel_linear(out)

            return out, dur_out, energy_out
        else:
            out, mel_pos, _ = self.variance_adaptor(
                enc_output,
                alpha=alpha,
                energy_alpha=energy_alpha,
            )
            out = self.decoder(out, mel_pos)
            out = self.mel_linear(out)
            return out