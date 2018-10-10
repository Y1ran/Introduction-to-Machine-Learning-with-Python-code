# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:00:01 2018

@author: Administrator
"""

import matplotlib.pyplot as plt

with open('data.csv') as f:
    x = []
    y = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        if len(lines) == 2:
            x.append(float(lines[0]))
            y.append(float(lines[1]))

plt.plot(x, y, 'b.', label="original data")
plt.title('Mean Shift')
plt.legend(loc="upper right")
plt.show()


import math
import sys
import numpy as np

MIN_DISTANCE = 0.000001#mini error

def load_data(path, feature_num=2):
    f = open(path)
    data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        if len(lines) != feature_num:
            continue
        for i in xrange(feature_num):
            data_tmp.append(float(lines[i]))

        data.append(data_tmp)
    f.close()
    return data

def gaussian_kernel(distance, bandwidth):
    m = np.shape(distance)[0]
    right = np.mat(np.zeros((m, 1)))
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))

    gaussian_val = left * right
    return gaussian_val

def shift_point(point, points, kernel_bandwidth):
    points = np.mat(points)
    m,n = np.shape(points)
    #计算距离
    point_distances = np.mat(np.zeros((m,1)))
    for i in range(m):
        point_distances[i, 0] = np.sqrt((point - points[i]) * (point - points[i]).T)

    #计算高斯核      
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)

    #计算分母
    all = 0.0
    for i in range(m):
        all += point_weights[i, 0]

    #均值偏移
    point_shifted = point_weights.T * points / all
    return point_shifted

def euclidean_dist(pointA, pointB):
    #计算pointA和pointB之间的欧式距离
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)

def distance_to_group(point, group):
    min_distance = 10000.0
    for pt in group:
        dist = euclidean_dist(point, pt)
        if dist < min_distance:
            min_distance = dist
    return min_distance

def group_points(mean_shift_points):
    group_assignment = []
    m,n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        print (item_1)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1

    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

            item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])

    return group_assignment

def train_mean_shift(points, kenel_bandwidth=2):
    #shift_points = np.array(points)
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iter = 0
    m, n = np.shape(mean_shift_points)
    need_shift = [True] * m

    #cal the mean shift vector
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iter += 1
        print ("iter : " + str(iter))
        for i in range(0, m):
            #判断每一个样本点是否需要计算偏置均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kenel_bandwidth)
            dist = euclidean_dist(p_new, p_new_start)

            if dist > max_min_dist:#record the max in all points
                max_min_dist = dist
            if dist < MIN_DISTANCE:#no need to move
                need_shift[i] = False

            mean_shift_points[i] = p_new
    #计算最终的group
    group = group_points(mean_shift_points)

    return np.mat(points), mean_shift_points, group

if __name__ == "__main__":
    #导入数据集
    path = "data.csv"
    data = load_data(path, 2)

    #训练，h=2
    points, shift_points, cluster = train_mean_shift(data, 2)

    for i in range(len(cluster) - 1):
        print ("%5.2f,%5.2f\t%5.2f,%5.2f\t%i" % 
               (points[i,0], points[i, 1], shift_points[i, 0], 
                shift_points[i, 1], cluster[i]))
    import matplotlib.pyplot as plt

    f = open("data.csv")
    cluster_x_0 = []
    cluster_x_1 = []
    cluster_x_2 = []
    cluster_y_0 = []
    cluster_y_1 = []
    cluster_y_2 = []
    center_x = []
    center_y = []
    center_dict = {}
    
    for line in f.readlines():
        lines = line.strip().split("\t")
        if len(lines) == 3:
            label = int(lines[2])
            if label == 0:
                data_1 = lines[0].strip().split(",")
                cluster_x_0.append(float(data_1[0]))
                cluster_y_0.append(float(data_1[1]))
                if label not in center_dict:
                    center_dict[label] = 1
                    data_2 = lines[1].strip().split(",")
                    center_x.append(float(data_2[0]))
                    center_y.append(float(data_2[1]))
            elif label == 1:
                data_1 = lines[0].strip().split(",")
                cluster_x_1.append(float(data_1[0]))
                cluster_y_1.append(float(data_1[1]))
                if label not in center_dict:
                    center_dict[label] = 1
                    data_2 = lines[1].strip().split(",")
                    center_x.append(float(data_2[0]))
                    center_y.append(float(data_2[1]))
            else:
                data_1 = lines[0].strip().split(",")
                cluster_x_2.append(float(data_1[0]))
                cluster_y_2.append(float(data_1[1]))
                if label not in center_dict:
                    center_dict[label] = 1
                    data_2 = lines[1].strip().split(",")
                    center_x.append(float(data_2[0]))
                    center_y.append(float(data_2[1]))    
    f.close()
    
    
    plt.plot(cluster_x_0, cluster_y_0, 'b.', label="cluster_0")
    plt.plot(cluster_x_1, cluster_y_1, 'g.', label="cluster_1")
    plt.plot(cluster_x_2, cluster_y_2, 'k.', label="cluster_2")
    plt.plot(center_x, center_y, 'r+', label="mean point")
    plt.title('Mean Shift 2')
    #plt.legend(loc="best")
    plt.show()