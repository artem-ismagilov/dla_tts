import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

from hw_tts.model import FastSpeech2

import waveglow
import text
import audio
import utils
from tqdm.auto import tqdm
import os
import argparse


def main(text_to_synthesize, speed, pitch, energy, output):
    c = torch.load('fast_speech_2_best.pth', map_location='cpu')
    model = FastSpeech2()
    model.load_state_dict(c['state_dict'])
    model.cuda().eval()
    WaveGlow = utils.get_WaveGlow()

    def synthesis(model, text, alpha=1.0, energy_alpha=1.0, pitch_alpha=1.0):
        text = np.stack([text])
        src_pos = np.array([i+1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).long().cuda()
        src_pos = torch.from_numpy(src_pos).long().cuda()

        with torch.no_grad():
            mel = model.forward(sequence, src_pos, alpha=alpha, energy_alpha=energy_alpha, pitch_alpha=pitch_alpha)
        mel = mel.contiguous().transpose(1, 2)

        waveglow.inference.inference(mel, WaveGlow, output)

    processed_text = text.text_to_sequence(text_to_synthesize, ['english_cleaners'])

    synthesis(model, processed_text, speed, energy, pitch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesize audio from text")
    parser.add_argument(
        "-t",
        "--text",
        default=None,
        type=str,
        help="text to synthesize",
    )
    parser.add_argument(
        "-s",
        "--speed",
        default=1.0,
        type=float,
        help="speed coefficient",
    )
    parser.add_argument(
        "-p",
        "--pitch",
        default=1.0,
        type=float,
        help="pitch coefficient",
    )
    parser.add_argument(
        "-e",
        "--energy",
        default=1.0,
        type=float,
        help="energy coefficient",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='result.wav',
        type=str,
        help="file to save synthesis results",
    )

    args = parser.parse_args()

    main(args.text, args.speed, args.pitch, args.energy, args.output)
