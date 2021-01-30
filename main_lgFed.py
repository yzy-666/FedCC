#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import pickle
import itertools
import pandas as pd
import numpy as np
import torch
from time import *

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img_local_all, test_img_avg_all, test_img_ensemble_all, test_img_local, test_img

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)

    #arg.load_fed
    assert(len(args.load_fed) > 0)

    base_save_dir = os.path.join(base_dir, 'lg/{}'.format(args.load_fed))
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir, exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    # dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    # with open(dict_save_path, 'rb') as handle:          #'rb'以二进制读模式打开user相关信息
    #     dict_users_train, dict_users_test = pickle.load(handle)

    begin_time = time()
    # build model
    net_glob = get_model(args)
    net_glob.train()

    # use load_fed to get a pre-trained federated model
    # fed_model_path = os.path.join(base_dir, 'fed/{}'.format(args.load_fed))
    # net_glob.load_state_dict(torch.load(fed_model_path))

    #定义参数w_glob_keys（为全局模型所拥有的层（key））
    total_num_layers = len(net_glob.weight_keys)        #总层数
    a = net_glob.weight_keys
    w_glob_keys = net_glob.weight_keys[total_num_layers - args.num_layers_keep:]
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))


    num_param_glob = 0
    num_param_local = 0
    for key in net_glob.state_dict().keys():
        num_param_local += net_glob.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += net_glob.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))

    # generate list of local models for each user
    net_local_list = []
    for user in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))

    acc_test_avg, loss_test_avg = test_img_avg_all(net_glob, net_local_list, args, dataset_test)
    acc_test_local_list, _ = test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=True)
    acc_test_local = acc_test_local_list.mean()

    # training
    results_save_path = os.path.join(base_save_dir, 'results.csv')
    results_columns = ['epoch', 'acc_test_local', 'acc_test_avg', 'best_acc_local', 'acc_test_ens_avg', 'acc_test_ens_maj']

    loss_train = []
    loss_test_list = []
    acc_test_list = []

    list_loss_train = []
    list_loss_train_2 = []
    list_acc_train = []
    list_loss_test = []
    list_acc_test = []


    best_iter = -1
    best_acc_local = -1
    best_acc_list = acc_test_local_list
    best_net_list = copy.deepcopy(net_local_list)
    fina_net_list = copy.deepcopy(net_local_list)

    results = []
    results.append(np.array([-1, acc_test_local, acc_test_avg, acc_test_local, None, None]))
    print('Round {:3d}, Acc (local): {:.2f}, Acc (avg): {:.2f}, Acc (local-best): {:.2f}'.format(
        -1, acc_test_local, acc_test_avg, acc_test_local))

    for iter in range(args.epochs):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, {}".format(iter, idxs_users))
        w_keys_epoch = w_glob_keys

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = net_local_list[idx]

            w_local, loss = local.train(net=net_local.to(args.device), lr=args.lr)
            loss_locals.append(copy.deepcopy(loss))

            # sum up weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_keys_epoch:
                    w_glob[k] += w_local[k]


        loss_avg = sum(loss_locals) / len(loss_locals)
        list_loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in w_keys_epoch:
            w_glob[k] = torch.div(w_glob[k], m)

        # copy weights to each local model
        for idx in range(args.num_users):
            net_local = net_local_list[idx]
            w_local = net_local.state_dict()
            for k in w_keys_epoch:
                w_local[k] = w_glob[k]
            net_local.load_state_dict(w_local)

        # find best local models from after round
        acc_test_local_list, _ = test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=True)
        for user in range(args.num_users):
            if acc_test_local_list[user] >= best_acc_list[user]:
                best_acc_list[user] = acc_test_local_list[user]
                best_net_list[user] = copy.deepcopy(net_local_list[user])

        # average best models for local test
        acc_test_avg, loss_test_avg, net_glob = test_img_avg_all(net_glob, best_net_list, args, dataset_test, return_net=True)


        # average global layers of the best local models
        best_net_list_avg = copy.deepcopy(best_net_list)
        w_glob = net_glob.state_dict()
        for net_local in best_net_list_avg:
            w_local = net_local.state_dict()
            for k in w_keys_epoch:
                w_local[k] = w_glob[k]
            net_local.load_state_dict(w_local)

        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        list_loss_train_2.append(loss_train)
        list_acc_train.append(acc_train)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        list_loss_test.append(loss_test)
        list_acc_test.append(acc_test)
        print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
            iter, loss_avg, loss_test, acc_test))

        acc_test_local,loss_test_local= test_img_local_all(best_net_list_avg, args, dataset_test, dict_users_test)
        acc_train_local, loss_train_local = test_img_local(best_net_list_avg, args, dataset_train, dict_users_test)
        list_loss_train_2.append(loss_train_local)
        list_acc_train.append(acc_train_local)
        list_loss_test.append(loss_test_local)
        list_acc_test.append(acc_test_local)
        if acc_test_local > best_acc_local:
            best_acc_local = acc_test_local
            best_iter = iter
            final_net_list = copy.deepcopy(best_net_list_avg)

        print('Round {:3d}, Acc (local): {:.2f}, Acc (avg): {:.2f}, Acc (local-best): {:.2f}, loss_train_avg: {:.3f}, loss_test_local: {:.3f}, acc_test_local:{:.2f}'.format(
            iter, acc_test_local, acc_test_avg, best_acc_local,loss_avg,loss_test_local, acc_test_local))


    acc_test_local, loss_test_local = test_img_local_all(final_net_list, args, dataset_test, dict_users_test)
    acc_test_avg, loss_test_avg = test_img_avg_all(net_glob, final_net_list, args, dataset_test)
    acc_test_ens_avg, loss_test, acc_test_ens_maj = test_img_ensemble_all(final_net_list, args, dataset_test)
    print('Best model, acc (local): {}, acc (avg): {}, acc (ens,avg): {}, acc (ens,maj): {}'.format(
        acc_test_local, acc_test_avg, acc_test_ens_avg, acc_test_ens_maj))

    end_time = time()
    print('time:', end_time - begin_time)

