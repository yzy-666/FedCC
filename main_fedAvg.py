#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time

import matplotlib

from models.test import test_img

matplotlib.use('Agg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, noniid, iid, mnist_noniid_pro_train
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from cluster.KMeans import k_means_pro



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        # root
        # download
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            print('mnist_iid')
        else:
            dict_users = mnist_noniid_pro_train(dataset_train, args.num_users)
            print('mnist_noniid_pro')


    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=False, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=False, transform=trans_cifar)

        if args.iid:
            dict_users = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
            print('cifar-iid')
        else:
            dict_users, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user,
                                                   rand_set_all=rand_set_all)
            print('LG_cifar-noniid')

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    list_loss_train = []
    list_loss_train_2 = []
    list_acc_train = []
    list_loss_test = []
    list_acc_test = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    start = time.time()
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), 20, replace=False)
        print("Round {}, idxs_users {}".format(iter, idxs_users))
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        list_loss_train.append(loss_avg)

        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        list_loss_train_2.append(loss_train)
        list_acc_train.append(acc_train)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        list_loss_test.append(loss_test)
        list_acc_test.append(acc_test)
        print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
            iter, loss_avg, loss_test, acc_test))
    end = time.time()
