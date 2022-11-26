import numpy as np

from text import text_to_sequence
import os
import audio

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import shutil
import pyworld as pw
import librosa


def build_from_path(in_dir, out_dir):
    index = 1
    # executor = ProcessPoolExecutor(max_workers=4)
    # futures = []
    texts = []

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            if index % 100 == 0:
                print("{:d} Done".format(index))
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            # futures.append(executor.submit(
            #     partial(_process_utterance, out_dir, index, wav_path, text)))
            texts.append(_process_utterance(out_dir, index, wav_path, text))

            index = index + 1

    # return [future.result() for future in tqdm(futures)]
    return texts


def _process_utterance(out_dir, index, wav_path, text):
    from audio.hparams_audio import hop_length

    # compute pitch
    wav, sr = librosa.load(wav_path)

    pitch, t = pw.dio(
        wav.astype(np.float64),
        sr,
        frame_period=hop_length / sr * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram, energy = audio.tools.get_mel(wav_path)
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)
    energy = energy.squeeze().numpy().astype(np.float32)

    assert (pitch.shape[0] == mel_spectrogram.shape[1]) and (energy.shape[0] == mel_spectrogram.shape[1])

    # Write the spectrograms to disk:
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    energy_filename = 'ljspeech-energy-%05d.npy' % index
    pitch_filename = 'ljspeech-pitch-%05d.npy' % index
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, energy_filename),
            energy, allow_pickle=False)
    np.save(os.path.join(out_dir, pitch_filename),
            pitch, allow_pickle=False)

    return text


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = 'mels'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)

    shutil.move(os.path.join(out_dir, "train.txt"),
                os.path.join("data", "train.txt"))


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')


def main():
    path = os.path.join("data", "LJSpeech-1.1")
    preprocess_ljspeech(path)


if __name__ == "__main__":
    main()
