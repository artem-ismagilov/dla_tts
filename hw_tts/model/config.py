import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    variance_embedding_bins = 256
    energy_min_max = (-1.1357148885726929, 15.167808532714844)
    pitch_min_max = (-1.1868396117310913, 6.822635708394187)

    variance_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1

    num_mels = 80

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
