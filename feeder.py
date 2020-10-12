import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os, glob



class GrabFeeder(torch.utils.data.Dataset):
    def __init__(self, data_path, selected_classes):
        self.data_path = data_path
        self.selected_classes = selected_classes
        self.length = 300
        self.load_data()

    def load_data(self):
        data = []
        class_label = []
        touching_points = []
        for i in range(len(self.selected_classes)):
            selected_class = self.selected_classes[i]
            class_files = glob.glob(self.data_path + '/*'+ selected_class +'*.npz')
            for class_file in class_files:
                poses, tp = load_sample_from_file(class_file, length=self.length)
                data.append(poses)
                class_label.append(i)
                touching_points.append(tp)

        self.class_label = class_label
        self.tp = np.array(touching_points)
        self.tp = self.tp.reshape(-1, int(self.length/4), 4, 18)
        self.tp = self.tp.max(axis=2)

        self.data = np.array(data)
        self.data = self.data.reshape((-1, 3, self.length, 18, 1))

    def __len__(self):
        teste = len(self.class_label)
        return teste

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.class_label[index]
        tp = self.tp[index]

        return data_numpy, tp, label


def load_sample_from_file(path, length=300):
    file_data = np.load(path)
    data = file_data['arr_0']
    data = np.transpose(data, (2, 0, 1))
    frames = data.shape[1]
    interval = frames/length
    idxs = np.floor(np.arange(length)*interval).astype(int)
    touching_points = data[3,idxs,:]
    pose_sequence = data[0:3,idxs,:]
    return pose_sequence, touching_points