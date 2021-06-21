import argparse
import os
import random
import shutil
import torch
import numpy as np
from tqdm import tqdm


def get_norm(file_path):
    with open(file_path, "r") as f:
        lines = [list(map(float, line[:-1].split(" ")))
                 for line in tqdm(f, desc="Loading Norm")]
    normalize_data = torch.tensor(lines)
    mean = normalize_data[0]
    std = normalize_data[1]
    for i in range(std.size(0)):
        if std[i] == 0:
            std[i] = 1
    return mean, std


def data_p(root_dir, type, out_dir):
    mean, std = get_norm(os.path.join(root_dir, type + "Norm.txt"))
    data_list = []
    file = open(os.path.join(root_dir, type + ".txt"), 'r')
    while True:
        data_str = file.readline().strip()
        if data_str == '' or data_str == '\n':
            break
        data = [[float(x) for x in data_str.split(' ')]]
        data = torch.tensor(data)
        if data.size(-1) == 5307 or data.size(-1) == 618:
            data = (data - mean) / std
            data_list.append(data)
    data = torch.cat(data_list, dim=0)
    torch.save(data, os.path.join(out_dir, type + '.pth'))
    print(out_dir + " data finish")


def data_preprocess_two_all(train_root,  output_root):
    train_dir = os.path.join(output_root, "Train")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_input_dir = os.path.join(train_dir, "Input")
    train_output_dir = os.path.join(train_dir, "Label")
    if not os.path.exists(train_input_dir):
        os.mkdir(train_input_dir)
    if not os.path.exists(train_output_dir):
        os.mkdir(train_output_dir)

    data_p(train_root, "Input", train_input_dir)
    data_p(train_root, "Output", train_output_dir)

    print("Preprocess Data Complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--train_root", type=str, help="train data file root dir")
    parser.add_argument("--output_root", type=str, help="output file root dir")
    args = parser.parse_args()
    data_preprocess_two_all(args.train_root, args.output_root)
