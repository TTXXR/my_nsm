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


def data_p(root_dir, pre, type, out_dir):
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
            data[:618] = (data[:618] - mean) / std
            data_list.append(data)
    data = torch.cat(data_list, dim=0)
    torch.save(data, os.path.join(out_dir, type + '.pth'))
    print(out_dir + " data finish")


def data_preprocess_two_all(root, type, output_root):
    dir = os.path.join(output_root, type)
    if not os.path.exists(dir):
        os.mkdir(dir)
    input_dir = os.path.join(dir, "Input")
    output_dir = os.path.join(dir, "Label")
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data_p(root, type+"_", "Input", input_dir)
    data_p(root, type+"_", "Output", output_dir)
    print("Preprocess Data Complete")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="data file root dir")
    parser.add_argument("--type", type=str, help="train/test")
    parser.add_argument("--output_root", type=str, help="output file root dir")
    args = parser.parse_args()
    data_preprocess_two_all(args.root, args.type, args.output_root)
