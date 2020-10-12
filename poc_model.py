import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os, glob

from stgcn import Model, Feeder

class PocModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        # Load layers from base model
        self.graph = base_model.graph
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)


        self.st_gcn_networks = base_model.st_gcn_networks
        self.data_bn = base_model.data_bn
        self.edge_importance = base_model.edge_importance

        # For activity recognition
        self.fcn = base_model.fcn

        # For point of contact prediction
        self.poc_conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=(1, 1))
        self.poc_conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=(1, 1))
        self.poc_conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=(1, 1))


    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # BRANCH 1 - Activity recognition
        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(N, M, -1, 1, 1).mean(dim=1)
        x1 = self.fcn(x1)
        x1 = x1.view(x1.size(0), -1)

        # BRANCH 2 - Point of contact prediction
        x2 = F.relu(self.poc_conv1(x))
        x2 = F.relu(self.poc_conv2(x2))
        x2 = F.sigmoid(self.poc_conv3(x2))
        x2 = x2.view(x2.size(0), x2.size(2), x2.size(3))

        return x1, x2