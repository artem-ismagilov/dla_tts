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

                mel_out, dur_out, energy_out = self.model(
                    batch['text'].to(self.device).long(),
                    batch['src_pos'].to(self.device).long(),
                    mel_pos=batch['mel_pos'].to(self.device).long(),
                    length_target=dur_target,
                    energy_target=energy_target,
                    mel_max_length=batch['mel_max_len'],
                )

                mel_loss, dur_loss, energy_loss = self.criterion(
                    mel_out,
                    dur_out,
                    energy_out,
                    target,
                    dur_target,
                    energy_target,
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

    def _log_audio(self, wave_batch):
        self.writer.add_audio("audio", random.choice(wave_batch.cpu()), self.config["preprocessing"]["sr"])

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

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
