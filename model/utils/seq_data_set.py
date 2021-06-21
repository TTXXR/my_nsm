import random

import torch
import torch.utils.data as tordata
import numpy as np
import os
from tqdm import tqdm


class DataSet(tordata.Dataset):

    def __init__(self, input_dir, label_dir, cache, ):
        self.input_data_dir = input_dir
        assert os.path.exists(self.input_data_dir), 'No Input dir'
        self.label_data_dir = label_dir
        assert os.path.exists(self.label_data_dir), 'No label dir'

        self.input_data = torch.load(os.path.join(self.input_data_dir, 'Input.pth'))
        self.label_data = torch.load(os.path.join(self.label_data_dir, 'Output.pth'))

        self.len = self.input_data.size(0)

        self.cache = cache

    def __len__(self):
        return self.len

    def load_data(self, index):
        return self.__getitem__(index)

    def __getitem__(self, item):
        return self.input_data[item], self.label_data[item]
