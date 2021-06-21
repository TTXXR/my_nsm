import torch


def base_loss(x, y):
    """
    Calculate the loss of the square difference of all data

    :param x: output from model
    :param y: label from database
    :param data_length: data length difference
    """
    loss = torch.mean(torch.pow((x - y), 2))
    return loss


def last_loss(x, y):
    """
    Calculate the loss of the square difference of the last digit of all data

    :param x: output from model
    :param y: label from database
    :param data_length: data length difference
    """
    loss = torch.mean(torch.pow((x[:, -1, :] - y[:, -1, :]), 2))
    return loss


def mean_loss(x, y):
    # import csv
    # csv_writer = csv.writer(open('test.csv', 'w', newline=""))
    # csv_writer.writerow([i for i in range(618)])
    # a = x.mean(dim=1)[0].cpu().detach().numpy().tolist()
    # b = y[:, -1, :][0].cpu().detach().numpy().tolist()
    # csv_writer.writerow(a)
    # csv_writer.writerow(b)
    # exit()
    loss = torch.mean(torch.pow((x.mean(dim=1) - y[:, -1, :]), 2))
    return loss


def weight_loss(x, y):
    x = x.mean(dim=1)
    y = y[:, -1, :]
    loss = torch.mean()
    return loss
