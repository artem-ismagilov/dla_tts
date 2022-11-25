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
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "duration_loss", "mel_loss", writer=self.writer
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

        batch_idx = 00
        for batches in tqdm(self.train_dataloader):
            for batch in tqdm(batches):
                self.optimizer.zero_grad()

                target = batch['mel_target'].to(self.device).float()
                dur_target = batch['duration'].to(self.device).int()

                mel_out, dur_out = self.model(
                    batch['text'].to(self.device).long(),
                    batch['src_pos'].to(self.device).long(),
                    mel_pos=batch['mel_pos'].to(self.device).long(),
                    length_target=dur_target,
                    mel_max_length=batch['mel_max_len'],
                )

                mel_loss, dur_loss = self.criterion(mel_out, dur_out, target, dur_target)
                loss = mel_loss + dur_loss

                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()

                self.train_metrics.update('loss', loss.detach())
                self.train_metrics.update('mel_loss', mel_loss.detach())
                self.train_metrics.update('duration_loss', dur_loss.detach())

                batch_idx += 1

                if batch_idx % self.log_step == 0:
                    if self.writer is not None:
                        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode="train")
                        if self.lr_scheduler is not None:
                            self.writer.add_scalar(
                                "learning_rate", self.lr_scheduler.get_last_lr()[0]
                            )

                        print(f'Loss: {loss.detach()}')

                if batch_idx == self.len_epoch:
                    break

        return log

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
