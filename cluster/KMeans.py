from collections import defaultdict
from random import uniform
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import random

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2

    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point.
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = float('inf')  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    """
    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    # 字典里key不存在时，返回为0
    # 保存每个纬度上的最大，最小值
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return assignments, dataset



def k_means_pro(dataset,k):
    result = k_means(dataset, k)

    # 分组后进行处理
    # 获取分组信息
    b = result[0]


    b_norepeat = []
    for s in b:
        if s not in b_norepeat:
            b_norepeat.append(s)

    # 将属于同一组的客户端放在一起,每一行代表同一个组的客户端
    list = []
    for i in b_norepeat:
        result = [idx for idx, val in enumerate(b) if val == i]
        list.append(result)

    return list



if __name__ == '__main__':

    points = np.array([
        [-0.0092588, 0.02575905, -0.02775564, -0.02684111, 0.02593129],
        [-0.00926681, 0.02575104, -0.02776365, -0.03215066,  0.02062173],
        [-0.00945463,  0.02556321, -0.02795147 , -0.03000744,  0.02276496],
        [-0.01060645,  0.02441141, -0.02910328 , -0.02979652,  0.02297588],
        [-0.00960597,  0.02541188, -0.0281028 , -0.02959208,  0.02318032],
        ])


    result = k_means(points, 3)

#分组后进行处理
    #获取分组信息
    b = result[0]

    b = [0, 0, 1, 1, 0, 1, 2, 2, 1, 0]

    b_norepeat = []
    for s in b:
        if s not in b_norepeat:
            b_norepeat.append(s)


    #将属于同一组的客户端放在一起
    list = []
    for i in b_norepeat:
        result = [idx for idx, val in enumerate(b) if val == i]
        list.append(result)


    print(list)
    res = []
    for a in list:
        res.append(random.choice(a))
    print(res)








    # for k, v in list(result):
    #     if k is 0:
    #         plt.scatter(v[0], v[1], c='b')
    #     elif k is 1:
    #         plt.scatter(v[0], v[1], c='r')
    #     else:
    #         plt.scatter(v[0], v[1], c='y')
    #
    #
    # plt.show()


