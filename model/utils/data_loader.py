import os
from .seq_data_set import DataSet


def load_data(data_root, cache=True):
    print("Loading cache")
    input_dir = os.path.join(data_root, 'Input')
    label_dir = os.path.join(data_root, 'Label')
    data_source = DataSet(input_dir, label_dir, cache)
    print("Loading finish")
    return data_source
