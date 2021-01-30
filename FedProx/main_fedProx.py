import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, datasets
from torchsummary import summary
from datetime import datetime, timedelta

from distribute_data import mnist_noniid_pro, noniid, iid
from model import MLP, CNNCifar
from option import args_parser
from train import training

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    transforms_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_data_train = datasets.MNIST('./data/mnist/', train=True, download=False, transform=transforms_mnist)
    mnist_data_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=transforms_mnist)

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    cifar_dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    # number of training rounds
    rounds = 100
    # client fraction
    C = 0.1
    # number of clients
    K = 100
    # number of training passes on local dataset for each roung
    E = 1
    # batch size
    batch_size = 10
    # learning Rate
    lr = 0.01
    # proximal term constant
    mu = 0.01
    # percentage of clients to have fewer than E epochs
    percentage = 0
    # target_test_accuracy
    target_test_accuracy = 99.0


    # mnist
    # data partition dictionary
    # iid_dict = mnist_noniid_pro(cifar_dataset_train, 100)
    dict_users = iid(mnist_data_train, args.num_users)
    print('mnist-iid')

    # load model
    model = MLP()
    print(model)

    mnist_mlp_trained = training(model, rounds, batch_size, lr, mnist_data_train, dict_users, mnist_data_test, C,
                                     K, E, mu, percentage, "my_NLP,noniid_pro", "orange", target_test_accuracy)


############################################################################################
    # cifar-10
    # data partition dictionary
    # dict_users= iid(cifar_dataset_train, args.num_users)
    # print('cifar-iid')
    # dict_users, rand_set_all = noniid(cifar_dataset_train, args.num_users, args.shard_per_user)
    # dict_users_test, rand_set_all = noniid(cifar_dataset_train, args.num_users, args.shard_per_user,
    #                                        rand_set_all=rand_set_all)
    # print('LG_cifar-noiid')

    # load model
    # model = CNNCifar(args=args).to(args.device)
    # print(model)

    # cifar_cnn_trained = training(model, rounds, batch_size, lr, cifar_dataset_train, dict_users, cifar_dataset_test, C,
    #                                  K, E, mu, percentage, "my_NLP,noniid_pro", "orange", target_test_accuracy)