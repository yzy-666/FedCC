#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])   #对w_local进行copy
    for k in w_avg.keys():              #共有w_avg.keys()层
        for i in range(1, len(w)):      #每层都将len(w)个客户端的第k层聚合
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
