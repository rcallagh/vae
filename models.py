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
        self._mlp.append(nn.ReLU())
        #hidden layers
        for iLayer in range(len(hidden_size) - 1):
            self._mlp.append(nn.Linear(hidden_size[iLayer], hidden_size[iLayer+1]))
            self._mlp.append(nn.ReLU())

        self._mlp.append(nn.Linear(hidden_size[-1], out_size))
        # self._mlp.append(nn.Sigmoid())

    def forward(self, x):
        return self._mlp(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss_func: Callable, optimiser_class: torch.optim.Optimizer, lr: float = 1e-3) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss_func = loss_func
        self.optimiser_class = optimiser_class
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)

        xhat = nn.functional.sigmoid(self.decoder(z))

        return xhat

    def prepare_batch(self, batch):
        #For now, just return the data, don't use the labels
        return batch[0].flatten(start_dim=1, end_dim=-1)

    def training_step(self, batch, batch_idx):
        x = self.prepare_batch(batch)

        xhat = self(x)

        loss = self.loss_func(x, xhat)

        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.prepare_batch(batch)

        xhat = self(x)

        loss = self.loss_func(x, xhat)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimiser_class(self.parameters(), lr=self.lr)
