import argparse

from config import conf
from model.initialization import initialization

parser = argparse.ArgumentParser(description='Train')
# parser.add_argument('--cache', default=True, help='cache: if set as TRUE all the training data will be loaded at once'
#                                                   ' before the training start. Default: TRUE')
parser.add_argument('test', default="test", help='if test model')
parser.add_argument('epoch', default="90", help='test model pre')
opt = parser.parse_args()

if __name__ == '__main__':
    model = initialization(conf, train=True)
    if opt.test == "test":
        print("Testing START")
        model.test(opt.epoch)
        print("Testing COMPLETE")
    elif opt.test == "train":
        print("Training START")
        model.train()
        print("Training COMPLETE")
    else:
        print("input train or test.")

