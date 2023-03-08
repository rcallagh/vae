#!/usr/bin/env python3

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: list[int]) -> None:
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
