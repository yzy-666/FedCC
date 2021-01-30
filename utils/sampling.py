#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
import torch
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_pro_train(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idx_0 = idxs[:6500]
    idx_1_9 = idxs[6500:]
    idx_1 = idxs[6000:12500]
    idx_2 = idxs[12000:18500]
    idx_3 = idxs[18000:24500]
    idx_4 = idxs[24000:30500]
    idx_5 = idxs[30000:36500]
    idx_6 = idxs[36000:42500]
    idx_7 = idxs[42000:48500]
    idx_8 = idxs[48000:54500]
    idx_9 = idxs[54000:]

    # divide and assign(这么做会有数据重复，因为没有减去已经被选的数据)
    for i in range(55):
        dict_users[i] = set(np.random.choice(idx_0, 600, replace=False))

    # for i in range(55,num_users):
    #     dict_users[i] = set(np.random.choice(idx_1_9, 600, replace=False))

    for i in range(55,60):
        dict_users[i] = set(np.random.choice(idx_1, 600, replace=False))

    for i in range(60,65):
        dict_users[i] = set(np.random.choice(idx_2, 600, replace=False))

    for i in range(65,70):
        dict_users[i] = set(np.random.choice(idx_3, 600, replace=False))

    for i in range(70,75):
        dict_users[i] = set(np.random.choice(idx_4, 600, replace=False))

    for i in range(75,80):
        dict_users[i] = set(np.random.choice(idx_5, 600, replace=False))

    for i in range(80,85):
        dict_users[i] = set(np.random.choice(idx_6, 600, replace=False))

    for i in range(85,90):
        dict_users[i] = set(np.random.choice(idx_7, 600, replace=False))

    for i in range(90,95):
        dict_users[i] = set(np.random.choice(idx_8, 600, replace=False))

    for i in range(95,100):
        dict_users[i] = set(np.random.choice(idx_9, 600, replace=False))

    return dict_users

def mnist_noniid_pro_test(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 100, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    idx_0 = idxs[:1080]
    # idx_1_9 = idxs[6500:]
    idx_1 = idxs[1000:2080]
    idx_2 = idxs[2000:3080]
    idx_3 = idxs[3000:4080]
    idx_4 = idxs[4000:5080]
    idx_5 = idxs[5000:6080]
    idx_6 = idxs[6000:7080]
    idx_7 = idxs[7000:8080]
    idx_8 = idxs[8000:9080]
    idx_9 = idxs[9000:]

    # divide and assign(这么做会有数据重复，因为没有减去已经被选的数据)
    for i in range(55):
        dict_users[i] = set(np.random.choice(idx_0, 100, replace=False))

    # for i in range(55,num_users):
    #     dict_users[i] = set(np.random.choice(idx_1_9, 600, replace=False))

    for i in range(55,60):
        dict_users[i] = set(np.random.choice(idx_1, 100, replace=False))

    for i in range(60,65):
        dict_users[i] = set(np.random.choice(idx_2, 100, replace=False))

    for i in range(65,70):
        dict_users[i] = set(np.random.choice(idx_3, 100, replace=False))

    for i in range(70,75):
        dict_users[i] = set(np.random.choice(idx_4, 100, replace=False))

    for i in range(75,80):
        dict_users[i] = set(np.random.choice(idx_5, 100, replace=False))

    for i in range(80,85):
        dict_users[i] = set(np.random.choice(idx_6, 100, replace=False))

    for i in range(85,90):
        dict_users[i] = set(np.random.choice(idx_7, 100, replace=False))

    for i in range(90,95):
        dict_users[i] = set(np.random.choice(idx_8, 100, replace=False))

    for i in range(95,100):
        dict_users[i] = set(np.random.choice(idx_9, 100, replace=False))

    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid_pro(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs_dict为一个字典，key未类别0,1...9，value为对应类别的图片下标0,1...49999
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    idx_0 = idxs_dict[0]
    idx_1 = idxs_dict[1]
    idx_2 = idxs_dict[2]
    idx_3 = idxs_dict[3]
    idx_4 = idxs_dict[4]
    idx_5 = idxs_dict[5]
    idx_6 = idxs_dict[6]
    idx_7 = idxs_dict[7]
    idx_8 = idxs_dict[8]
    idx_9 = idxs_dict[9]
    idx_1_9 = idx_1+idx_2+idx_3+idx_4+idx_5+idx_6+idx_7+idx_8+idx_9


    # divide and assign(这么做会有数据重复，因为没有减去已经被选的数据)
    for i in range(55):
        a = set(np.random.choice(idx_0, 455, replace=False))
        b = set(np.random.choice(idx_1, 45, replace=False))
        dict_users[i] = a | b

    for i in range(55, 60):
        a = set(np.random.choice(idx_1, 455, replace=False))
        b = set(np.random.choice(idx_2, 45, replace=False))
        dict_users[i] = a|b

    for i in range(60, 65):
        a = set(np.random.choice(idx_2, 455, replace=False))
        b = set(np.random.choice(idx_3, 45, replace=False))
        dict_users[i] = a | b

    for i in range(65, 70):
        a = set(np.random.choice(idx_3, 455, replace=False))
        b = set(np.random.choice(idx_4, 45, replace=False))
        dict_users[i] = a | b

    for i in range(70, 75):
        a = set(np.random.choice(idx_4, 455, replace=False))
        b = set(np.random.choice(idx_5, 45, replace=False))
        dict_users[i] = a | b

    for i in range(75, 80):
        a = set(np.random.choice(idx_5, 455, replace=False))
        b = set(np.random.choice(idx_6, 45, replace=False))
        dict_users[i] = a | b

    for i in range(80, 85):
        a = set(np.random.choice(idx_6, 455, replace=False))
        b = set(np.random.choice(idx_7, 45, replace=False))
        dict_users[i] = a | b

    for i in range(85, 90):
        a = set(np.random.choice(idx_7, 455, replace=False))
        b = set(np.random.choice(idx_8, 45, replace=False))
        dict_users[i] = a | b

    for i in range(90, 95):
        a = set(np.random.choice(idx_8, 455, replace=False))
        b = set(np.random.choice(idx_9, 45, replace=False))
        dict_users[i] = a | b

    for i in range(95, 100):
        a = set(np.random.choice(idx_9, 455, replace=False))
        b = set(np.random.choice(idx_8, 45, replace=False))
        dict_users[i] = a | b



    return dict_users

def cifar_noniid_pro_test(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # idxs_dict为一个字典，key未类别0,1...9，value为对应类别的图片下标0,1...49999
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    idx_0 = idxs_dict[0]
    idx_1 = idxs_dict[1]
    idx_2 = idxs_dict[2]
    idx_3 = idxs_dict[3]
    idx_4 = idxs_dict[4]
    idx_5 = idxs_dict[5]
    idx_6 = idxs_dict[6]
    idx_7 = idxs_dict[7]
    idx_8 = idxs_dict[8]
    idx_9 = idxs_dict[9]
    idx_1_9 = idx_1+idx_2+idx_3+idx_4+idx_5+idx_6+idx_7+idx_8+idx_9


    # divide and assign(这么做会有数据重复，因为没有减去已经被选的数据)
    for i in range(55):
        a = set(np.random.choice(idx_0, 90, replace=False))
        b = set(np.random.choice(idx_1, 10, replace=False))
        dict_users[i] = a | b

    for i in range(55, 60):
        a = set(np.random.choice(idx_1, 90, replace=False))
        b = set(np.random.choice(idx_2, 10, replace=False))
        dict_users[i] = a|b

    for i in range(60, 65):
        a = set(np.random.choice(idx_2, 90, replace=False))
        b = set(np.random.choice(idx_3, 10, replace=False))
        dict_users[i] = a | b

    for i in range(65, 70):
        a = set(np.random.choice(idx_3, 90, replace=False))
        b = set(np.random.choice(idx_4, 10, replace=False))
        dict_users[i] = a | b

    for i in range(70, 75):
        a = set(np.random.choice(idx_4, 90, replace=False))
        b = set(np.random.choice(idx_5, 10, replace=False))
        dict_users[i] = a | b

    for i in range(75, 80):
        a = set(np.random.choice(idx_5, 90, replace=False))
        b = set(np.random.choice(idx_6, 10, replace=False))
        dict_users[i] = a | b

    for i in range(80, 85):
        a = set(np.random.choice(idx_6, 90, replace=False))
        b = set(np.random.choice(idx_7, 10, replace=False))
        dict_users[i] = a | b

    for i in range(85, 90):
        a = set(np.random.choice(idx_7, 90, replace=False))
        b = set(np.random.choice(idx_8, 10, replace=False))
        dict_users[i] = a | b

    for i in range(90, 95):
        a = set(np.random.choice(idx_8, 90, replace=False))
        b = set(np.random.choice(idx_9, 10, replace=False))
        dict_users[i] = a | b

    for i in range(95, 100):
        a = set(np.random.choice(idx_9, 90, replace=False))
        b = set(np.random.choice(idx_8, 10, replace=False))
        dict_users[i] = a | b



    return dict_users

def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users




if __name__ == '__main__':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=trans_mnist)

    num = 100
    d = mnist_noniid_pro_test(dataset_test, num)
    a = 1
