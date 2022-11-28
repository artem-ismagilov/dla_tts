import pathlib
import random
import itertools

from IPython import display
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import distributions
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import MelSpectrogram
import math
import time
import os
import librosa
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from dataclasses import dataclass
from collections import OrderedDict

from text import text_to_sequence


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer():
    buffer = list()
    text = process_text("./data/train.txt")

    energy_min, energy_max = 1000, 0
    energy_std, energy_mean = [], []

    pitch_min, pitch_max = 1000, 0
    pitch_std, pitch_mean = [], []

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join("./mels", "ljspeech-mel-%05d.npy" % (i+1))
        energy_gt_name = os.path.join("./mels", "ljspeech-energy-%05d.npy" % (i+1))
        pitch_gt_name = os.path.join("./mels", "ljspeech-pitch-%05d.npy" % (i+1))

        mel_gt_target = np.load(mel_gt_name)
        energy_gt_target = np.load(energy_gt_name)
        pitch_gt_target = np.load(pitch_gt_name)

        duration = np.load(os.path.join("./alignments", str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(text_to_sequence(character, ['english_cleaners']))

        energy_min = min(energy_min, np.amin(energy_gt_target))
        energy_max = max(energy_max, np.amax(energy_gt_target))
        energy_std.append(np.std(energy_gt_target))
        energy_mean.append(np.mean(energy_gt_target))

        pitch_min = min(pitch_min, np.amin(pitch_gt_target))
        pitch_max = max(pitch_max, np.amax(pitch_gt_target))
        pitch_std.append(np.std(pitch_gt_target))
        pitch_mean.append(np.mean(pitch_gt_target))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        energy_gt_target = torch.from_numpy(energy_gt_target)
        pitch_gt_target = torch.from_numpy(pitch_gt_target)

        buffer.append({
            "text": character,
            "duration": duration,
            "mel_target": mel_gt_target,
            "energy": energy_gt_target,
            "pitch": pitch_gt_target,
        })

    energy_std = np.mean(energy_std)
    energy_mean = np.mean(energy_mean)
    energy_max = (energy_max - energy_mean) / energy_std
    energy_min = (energy_min - energy_mean) / energy_std

    pitch_std = np.mean(pitch_std)
    pitch_mean = np.mean(pitch_mean)
    pitch_max = (pitch_max - pitch_mean) / pitch_std
    pitch_min = (pitch_min - pitch_mean) / pitch_std

    for b in buffer:
        b['energy'] = (b['energy'] - energy_mean) / energy_std
        b['pitch'] = (b['pitch'] - pitch_mean) / pitch_std

    end = time.perf_counter()

    print(f'energy bounds: {energy_min} {energy_max}')
    print(f'pitch bounds: {pitch_min} {pitch_max}')
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class LJSpeechDataset(Dataset):
    def __init__(self):
        self.buffer = get_data_to_buffer()
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output

def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    energy = [batch[ind]["energy"] for ind in cut_list]
    pitch = [batch[ind]["pitch"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    energy = pad_1D_tensor(energy)
    pitch = pad_1D_tensor(pitch)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "energy": energy,
           "pitch": pitch,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out


class Collator:
    def __init__(self, batch_size, batch_expand_size):
        self.batch_size = batch_size
        self.batch_expand_size = batch_expand_size

    def __call__(self, batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // self.batch_expand_size

        cut_list = list()
        for i in range(self.batch_expand_size):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(self.batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output


class LJSpeechDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        batch_expand_size,
        num_workers
    ):
        super().__init__(
            dataset,
            collate_fn=Collator(batch_size, batch_expand_size),
            batch_size=batch_expand_size * batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

