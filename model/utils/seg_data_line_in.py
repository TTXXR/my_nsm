from tqdm import tqdm
import pandas as pd
import argparse


def _seg(ori, obj):
    label_data = pd.read_csv(ori + "/Output.txt", sep=' ', header=None, dtype=float)
    print("Reading Finished.")
    train_num = int(len(label_data) * 0.8)

    f_input = open(ori + "Input.txt")
    f_label = open(ori + "Output.txt")

    f_train_input = open(obj + "Train_Input.txt", 'w')
    f_train_label = open(obj + "Train_Output.txt", 'w')
    f_test_input = open(obj + "Test_Input.txt", 'w')
    f_test_label = open(obj + "Test_Output.txt", 'w')

    # input
    n = 0
    input_line = f_input.readline()
    f_train_input.write(input_line)
    n += 1
    while input_line:
        input_line = f_input.readline()
        if n < train_num:
            f_train_input.write(input_line)
        else:
            f_test_input.write(input_line)
        n += 1
    f_input.close()
    f_train_input.close()
    f_test_input.close()
    print("Seg Input Finished.")

    # label
    nn = 0
    label_line = f_label.readline()
    f_train_label.write(label_line)
    nn += 1
    while label_line:
        label_line = f_label.readline()
        if nn < train_num:
            f_train_label.write(label_line)
        else:
            f_test_label.write(label_line)
        nn += 1
    f_label.close()
    f_train_label.close()
    f_test_label.close()
    print("Seg Label Finished.")


if __name__ == '__main__':
    # "F:\AI4Animation-master\AI4Animation\SIGGRAPH_Asia_2019\Export"
    parser = argparse.ArgumentParser()
    parser.add_argument("origin", type=str, help="")
    parser.add_argument("object", type=str, help="")
    args = parser.parse_args()
    _seg(args.origin, args.object)
