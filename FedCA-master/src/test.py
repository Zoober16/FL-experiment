# -*- coding: utf-8 -*-
# Python version: 3.11
"""
Created on 12/12/2023

@author: junliu
"""

import copy

def normalize_list_values(list_orig, max_value=1):

    # 确保列表不为空
    if not list_orig:
        return []

    min_orig = min(list_orig) / 1.1
    max_orig = max(list_orig)

    # 规范化到 [0, 1]
    normalized = [(item - min_orig) / (max_orig - min_orig) if max_orig != min_orig else 1.0 for item in list_orig]

    # 计算调整因子，使得最小值大于0
    min_adjustment = 1.0 / len(list_orig)
    # min_adjustment = min_orig / sum(list_orig)
    adjusted = [item + min_adjustment for item in normalized]

    # 调整总和为 max_value
    sum_adjusted = sum(adjusted)
    final_list = [item * max_value / sum_adjusted for item in adjusted]

    return final_list


def normalize1_list_values(list_orig):

    if not list_orig:
        return []

    if sum(list_orig) != 0:
        sum_orig = sum(list_orig)
        list_normal = [item / sum_orig for item in list_orig]
    else:
        list_normal = [0.0 for i in range(len(list_orig))]

    return list_normal

if __name__ == '__main__':

    list0_to_normalize = [
        0.9739743024110794, 0.9772621765732765, 0.9751173555850983, 0.9761251881718636,
        0.9737694263458252, 0.9740959629416466, 0.9763607680797577, 0.9776016920804977,
        0.7528989056125283, 0.756142117548734
    ]
    list1_to_normalize = [
        0.9739743024110794, 0.9772621765732765, 0.9751173555850983, 0.9761251881718636,
        0.9737694263458252, 0.9740959629416466, 0.9763607680797577, 0.9776016920804977,
        0.8528989056125283, 0.856142117548734
    ]
    list2_to_normalize = [
        0.9739743024110794, 0.9772621765732765, 0.9751173555850983, 0.9761251881718636,
        0.9737694263458252, 0.9740959629416466, 0.9763607680797577, 0.9776016920804977,
        0.9528989056125283, 0.956142117548734
    ]

    # new1_normalized_list = normalize1_list_values(list1_to_normalize)
    # new2_normalized_list = normalize1_list_values(list2_to_normalize)
    new0_normalized_list = normalize_list_values(list0_to_normalize, 1)
    new1_normalized_list = normalize_list_values(list1_to_normalize, 1)
    new2_normalized_list = normalize_list_values(list2_to_normalize, 1)
    print('new0_normalized_list:\n{}'.format(new0_normalized_list))
    print('new1_normalized_list:\n{}'.format(new1_normalized_list))
    print('new2_normalized_list:\n{}'.format(new2_normalized_list))
