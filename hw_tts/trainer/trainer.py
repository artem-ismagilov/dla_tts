import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker

import waveglow
import text
import audio
import utils
import os
import numpy as np
import librosa
import time


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            dataloader,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = 15

        self.train_metrics = MetricTracker(
            "loss", "duration_loss", "mel_loss", "energy_loss", writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            if tensor_for_gpu not in batch:
                continue
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        print('LR:', self.lr_scheduler.get_last_lr()[0])

        batch_idx = 0
        for batches in tqdm(self.train_dataloader):
            for batch in batches:
                self.optimizer.zero_grad()

                target = batch['mel_target'].to(self.device).float()
                dur_target = batch['duration'].to(self.device).int()
                energy_target = batch['energy'].to(self.device).float()
                pitch_target = batch['pitch'].to(self.device).float()

                mel_out, dur_out, energy_out, pitch_out = self.model(
                    batch['text'].to(self.device).long(),
                    batch['src_pos'].to(self.device).long(),
                    mel_pos=batch['mel_pos'].to(self.device).long(),
                    length_target=dur_target,
                    energy_target=energy_target,
                    pitch_target=pitch_target,
                    mel_max_length=batch['mel_max_len'],
                )

                mel_loss, dur_loss, energy_loss, pitch_loss = self.criterion(
                    mel_out,
                    dur_out,
                    energy_out,
                    pitch_out,
                    target,
                    dur_target,
                    energy_target,
                    pitch_target,
                )

                loss = mel_loss + dur_loss + energy_loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()

                self.train_metrics.update('loss', loss.detach())
                self.train_metrics.update('mel_loss', mel_loss.detach())
                self.train_metrics.update('duration_loss', dur_loss.detach())
                self.train_metrics.update('energy_loss', energy_loss.detach())

                batch_idx += 1

                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), loss.item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()

            if batch_idx >= self.len_epoch:
                break

        self._log_synthesis()

        log = last_train_metrics
        return log

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, label, wave, sr):
        self.writer.add_audio(label, torch.tensor(wave), sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_synthesis(self):
        print('Synthesize test audios...')
        start = time.time()

        os.makedirs("results", exist_ok=True)
        WaveGlow = utils.get_WaveGlow()
        self.model.eval()

        data = self.get_test_data()

        for speed in [0.8, 1, 1.2]:
            for i, t in enumerate(data):
                self.synthesis(self.model, WaveGlow, t, f'results/{i}_speed_{speed}.wav', alpha=speed)

        for energy in [0.8, 1, 1.2]:
            for i, t in enumerate(data):
                self.synthesis(self.model, WaveGlow, t, f'results/{i}_energy_{energy}.wav', energy_alpha=energy)

        for pitch in [0.8, 1, 1.2]:
            for i, t in enumerate(data):
                self.synthesis(self.model, WaveGlow, t, f'results/{i}_energy_{pitch}.wav', pitch_alpha=pitch)

        for k in [0.8, 1, 1.2]:
            for i, t in enumerate(data):
                self.synthesis(
                    self.model,
                    WaveGlow,
                    t,
                    f'results/{i}_all_{k}.wav',
                    alpha=k,
                    energy_alpha=k,
                    pitch_alpha=k,)

        for f in os.listdir('results'):
            wav, sr = librosa.load(os.path.join('results', f))
            self._log_audio(f, wav, sr)

        print(f'Synthesize took {time.time() - start} seconds')

    @staticmethod
    def synthesis(model, WaveGlow, text, fout, alpha=1.0, energy_alpha=1.0, pitch_alpha=1.0):
        text = np.stack([text])
        src_pos = np.array([i+1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).long().cuda()
        src_pos = torch.from_numpy(src_pos).long().cuda()

        with torch.no_grad():
            mel = model.forward(sequence, src_pos, alpha=alpha, energy_alpha=energy_alpha, pitch_alpha=1.0)
        mel = mel.contiguous().transpose(1, 2)
        waveglow.inference.inference(mel, WaveGlow, fout)

    @staticmethod
    def get_test_data():
        tests = [
            "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest.",
            "Massachusetts Institute of Technology may be best known for its math, science and engineering education.",
            "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space."
        ]
        data_list = list(text.text_to_sequence(test, ['english_cleaners']) for test in tests)

        return data_list

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
