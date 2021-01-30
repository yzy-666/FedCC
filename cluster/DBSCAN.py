import random

import numpy as np
from scipy.spatial import KDTree
from sklearn import datasets


def dbscan2(data, eps, minpts):
    kdtree = KDTree(data)
    n = data.shape[0]  # 样本数
    k = 0  # 类编号
    visit = np.zeros(n)  # 是否被访问过
    res = np.zeros(n)  # 聚类结果
    random_id = np.random.permutation(n)  # 随机排列
    for s in random_id:
        if visit[s] == 0:
            visit[s] = 1
            neps = list(kdtree.query_radius(data[s].reshape(1, -1), eps)[0])  # 找到 eps 范围邻域内所有点(包括了自己)
            if len(neps)-1 < minpts:  # 数量不足 minpts 暂时设为噪声点
                res[s] = -1
            else:
                k += 1
                res[s] = k  # 数量达到 minpts 归为 k 类
                while len(neps) > 0:
                    j = random.choice(neps)
                    neps.remove(j)
                    if res[j] == -1:  # 如果之前被标为噪声点，则归为该类
                        res[j] = k
                    if visit[j] == 0:  # 没有被访问过
                        visit[j] = 1
                        j_neps = list(kdtree.query_radius(data[j].reshape(1, -1), eps)[0])  # 找邻域
                        if len(j_neps)-1 < minpts:
                            res[j] = k  # 非核心点，可归为该类, 也可以归为边界点
                        else:
                            res[j] = k  # 核心点，加入集合
                            neps = list(set(neps + j_neps))
    return res, k

if __name__ == '__main__':
    x1, y1 = datasets.make_circles(n_samples=6000, factor=0.5, noise=0.03)
    x2, y2 = datasets.make_blobs(n_samples=1000, centers=[[0, 0]], cluster_std=[0.085])
    X = np.vstack((x1, x2))
