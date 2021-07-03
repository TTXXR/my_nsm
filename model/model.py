import logging
import os
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.cpp_extension
import torch.utils.data as tordata

from .mlp_network import Encoder, Decoder

# Check GPU available
print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)
print('torch cuda version:', torch.version.cuda)
print('cuda is available:', torch.cuda.is_available())


class Model(object):
    def __init__(self,
                 # For Model base information
                 model_name, epoch, batch_size, save_path, load_path,
                 # For Date information
                 train_source, test_source,
                 # For encoder mlp_network information
                 encoder_dim, mlp_ratio, encoder_dropout,
                 # For decoder mlp_network information
                 decoder_dim, decoder_dropout,
                 # optim param
                 lr,
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.save_path = save_path
        self.load_path = load_path

        self.train_source = train_source
        self.test_source = test_source

        self.encoder_dim = encoder_dim
        self.mlp_ratio = mlp_ratio
        self.encoder_dropout = encoder_dropout
        self.decoder_dim = decoder_dim
        self.decoder_dropout = decoder_dropout

        # build mlp_network
        encoder = Encoder(self.encoder_dim, self.mlp_ratio, self.encoder_dropout)
        if torch.cuda.is_available():
            encoder = nn.DataParallel(encoder.cuda())
        self.encoder = encoder

        decoder = Decoder(self.decoder_dim, self.decoder_dropout)
        if torch.cuda.is_available():
            decoder = nn.DataParallel(decoder.cuda())
        self.decoder = decoder

        # build optimizer
        self.lr = lr
        self.encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.lr)

        # build loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

    def up_lr(self):
        self.lr /= 2
        for param_group in self.encoder_optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.decoder_optimizer.param_groups:
            param_group['lr'] = self.lr

    def save(self, e):
        # Save Model
        torch.save(self.encoder.state_dict(), os.path.join(self.save_path, str(e)+"encoder.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_path, str(e)+"decoder.pth"))
        # Save optimizer
        torch.save(self.encoder_optimizer.state_dict(), os.path.join(self.save_path, str(e)+"encoder_optimizer.pth"))
        torch.save(self.decoder_optimizer.state_dict(),
                   os.path.join(self.save_path, str(e)+"decoder_optimizer.pth"))

    def load(self, e):
        self.encoder.load_state_dict(torch.load(os.path.join(self.load_path, str(e)+'encoder.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(self.load_path, str(e)+'decoder.pth')))

        self.encoder_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, str(e)+'encoder_optimizer.pth')))
        self.decoder_optimizer.load_state_dict(
            torch.load(os.path.join(self.load_path, str(e)+'decoder_optimizer.pth')))

    def train(self):
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )
        self.encoder.train()
        self.decoder.train()

        train_loss = []
        for e in range(self.epoch):
            loss_list = []
            # learning rate update
            if (e+1)%30 == 0:
                self.up_lr()
            for x, y in tqdm(train_loader, ncols=100):
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                # Encoder Network
                x = self.encoder(x)
                # Decoder Network
                output = self.decoder(x)
                # loss
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = self.loss_function(output, y)
                loss_list.append(loss.item())

                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

            if e % 30 == 0 and e != 0:
                self.save(e)

            avg_loss = np.asarray(loss_list).mean()
            train_loss.append(avg_loss)
            print('Time {} '.format(datetime.datetime.now()),
                  'Epoch {} : '.format(e + 1),
                  'Training Loss = {:.9f} '.format(avg_loss),
                  'lr = {} '.format(self.lr),
                  )
            logging.info('Epoch {} : '.format(e + 1) +
                         'Training Loss = {:.9f} '.format(avg_loss) +
                         'lr = {} '.format(self.lr))
        torch.save(train_loss, os.path.join(self.save_path, 'trainloss.bin'))
        print('Learning Finished')

    def test(self):
        epoch = 30
        self.load(epoch)
        train_loader = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )
        self.encoder.eval()
        self.decoder.eval()

        test_loss = []
        for x, y in tqdm(train_loader, ncols=100):
            # Encoder Network
            x = self.encoder(x)
            # Decoder Network
            output = self.decoder(x)
            # loss
            if torch.cuda.is_available():
                y = y.cuda()
            loss = self.loss_function(output, y)
            test_loss.append(loss.item())

        avg_loss = np.asarray(test_loss).mean()
        print('Testing Loss = {:.9f} '.format(avg_loss))
        print('Testing Finished')
