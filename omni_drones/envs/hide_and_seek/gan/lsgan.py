import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
import pdb

def batch_feed_array(array, batch_size):
    data_size = array.shape[0]
    #assert data_size >= batch_size
    
    if data_size <= batch_size:
        while True:
            yield array
    else:
        start = 0
        while True:
            if start + batch_size < data_size:
                yield array[start:start + batch_size, ...]
            else:
                yield np.concatenate(
                    [array[start:data_size], array[0: start + batch_size - data_size]],
                    axis=0
                )
            start = (start + batch_size) % data_size

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self, gan_configs):
        super(Discriminator, self).__init__()
        self.input_size = gan_configs['goal_size']
        self.d_net = nn.Sequential(nn.Linear(self.input_size, 128), nn.LeakyReLU(0.2),
                                nn.Linear(128, 128), nn.LeakyReLU(0.2),
                                nn.Linear(128, gan_configs['num_labels']))
        for layer in self.d_net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        out = self.d_net(input)
        return out

class Generator(nn.Module):
    def __init__(self, gan_configs):
        super(Generator, self).__init__()
        self.input_size = gan_configs['gan_noise_size']
        self.g_net = nn.Sequential(nn.Linear(self.input_size, 256), 
                                   nn.ReLU(),
                                   nn.Linear(256, 256), 
                                   nn.ReLU(),
                                   nn.Linear(256, gan_configs['goal_size']),
                                   nn.Tanh())
        for layer in self.g_net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)


    def forward(self, z):
        out = self.g_net(z)
        return out

class LSGAN(object):
    def __init__(self, gan_configs):
        self.generator_output_size = gan_configs['goal_size']
        self.discriminator_output_size = gan_configs['num_labels']
        self.noise_size = gan_configs['gan_noise_size']
        self.generator = Generator(gan_configs)
        self.discriminator = Discriminator(gan_configs)
        # self.sample_discriminator = Discriminator(gan_configs)
        # self.generator_discriminator = Discriminator(gan_configs)
        self.adversarial_loss = torch.nn.MSELoss()
        self.gan_configs = gan_configs
        if gan_configs['cuda']:
            self.generator.cuda()
            self.discriminator.cuda()
            # self.sample_discriminator.cuda()
            # self.generator_discriminator.cuda()
            self.adversarial_loss.cuda()

        # self.generator.apply(weights_init_normal)
        # self.discriminator.apply(weights_init_normal)   
         
        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=gan_configs['lr'])
        # self.optimizer_D = torch.optim.RMSprop([{'params':self.sample_discriminator.parameters()},
        #                                         {'params':self.generator_discriminator.parameters()}], lr=gan_configs['lr'])
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=gan_configs['lr'])
        self.Tensor = torch.cuda.FloatTensor if gan_configs['cuda'] else torch.FloatTensor
    
    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def sample_generator(self, size):
        generator_samples = []
        generator_noise = []
        batch_size = self.gan_configs['batch_size']
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = self.Tensor(self.sample_random_noise(sample_size))
            generator_noise.append(noise)
            generator_samples.append(self.generator(noise))
        return torch.cat(generator_samples, dim=0), torch.cat(generator_noise, dim=0)

    def train(self, X, Y, gan_configs):
        batch_size = gan_configs['batch_size']
        batch_feed_X = batch_feed_array(X, batch_size)
        batch_feed_Y = batch_feed_array(Y, batch_size)
        generated_Y = self.Tensor(np.zeros((batch_size, self.discriminator_output_size)))
        for epoch in range(self.gan_configs['gan_outer_iters']):
            sample_X = self.Tensor(next(batch_feed_X))
            sample_Y = self.Tensor(next(batch_feed_Y))
            generated_X, random_noise = self.sample_generator(batch_size)
            generated_X_copy = generated_X.detach()
            train_X = torch.cat((sample_X, generated_X_copy), dim=0)
            train_Y = torch.cat((sample_Y, generated_Y), dim=0)
            
            # Train Discriminator
            sample_output = self.discriminator(train_X)
            self.optimizer_D.zero_grad()
            discriminator_loss = torch.mean((2*train_Y - 1 - sample_output)**2)
            discriminator_loss.backward()
            self.optimizer_D.step()
            # Train Generator
            generator_output = self.discriminator(generated_X)
            self.optimizer_G.zero_grad()
            generator_loss = torch.mean((generator_output - 1)**2)
            generator_loss.backward()
            self.optimizer_G.step()
        return discriminator_loss, generator_loss