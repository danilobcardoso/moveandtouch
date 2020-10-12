import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import numpy as np
import os, glob

from stgcn import Model, Feeder
from poc_model import PocModel
from feeder import GrabFeeder

in_channels = 3
num_class = 400
edge_importance_weighting = True
graph_args = {
    "layout": "openpose",
    "strategy": "spatial"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

stgcn_original_model = Model(in_channels, num_class, graph_args, edge_importance_weighting)

stgcn_state_dict = torch.load('./models/stgcn/st_gcn.kinetics-6fa43f73.pth')
stgcn_original_model.load_state_dict(stgcn_state_dict)

new_model = PocModel(stgcn_original_model)
new_model = new_model.to(device)

selected_classes = ['offhand', 'eat', 'drink']
grab_feeder = GrabFeeder('../../datasets/grab_skeleton/', selected_classes)

grab_data_loader = DataLoader(grab_feeder, batch_size=5, shuffle=True)

cross_entropy = nn.CrossEntropyLoss()
bce = nn.BCELoss()

optimizer = optim.SGD(
                new_model.parameters(),
                lr=0.01,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0001)

new_model.train()
loss_value = []
iter_info = {}

epochs = 20

writer = SummaryWriter('runs/grab_1')

# device = torch.device('cuda')

for epoch in tqdm(range(epochs)):
    for data, tp, label in grab_data_loader:

        data = data.float().to(device)
        label = label.long().to(device)
        tp = tp.float().to(device)

        # forward
        out1, out2 = new_model(data)
        loss1 = cross_entropy(out1, label)
        loss2 = bce(out2, tp)
        loss = loss1 + loss2

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        iter_info['loss'] = loss.data.item()
        iter_info['lr'] = '{:.6f}'.format(0.01)
        loss_value.append(iter_info['loss'])
        writer.add_scalar('training loss',  iter_info['loss'],  epoch )
        print(iter_info['loss'])