#!/usr/bin/env python3

from typing import Callable
import torch
import torch.nn as nn
import pytorch_lightning as pl

class MLP(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: "list[int]") -> None:
        super().__init__()

        assert len(hidden_size) > 0

        self._mlp = nn.Sequential()
        #input layer
        self._mlp.append(nn.Linear(in_size, hidden_size[0]))
        #hidden layers
        for iLayer in range(len(hidden_size) - 1):
            self._mlp.append(nn.Linear(hidden_size[iLayer], hidden_size[iLayer+1]))

        self._mlp.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, x):
        return self._mlp(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss_func: Callable, optimiser_class: torch.optim.Optimizer) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func
        self.optimser_class = optimiser_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)

        xhat = self.decoder(z)

        return xhat

    def prepare_batch(self, batch):
        return batch

    def training_step(self, batch, batch_idx):
        x = self.prepare_batch(batch)

        xhat = self(x)

        loss = self.loss_func(x, xhat)

        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimiser_class(self.parameters())
