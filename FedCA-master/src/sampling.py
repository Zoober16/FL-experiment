# -*- coding: utf-8 -*-
# Python version: 3.11
"""
Created on 12/12/2023

@author: junliu
"""

import numpy as np

import skimage
import torchvision
import random
import torch
from torchvision import datasets, transforms


def specical_processing(dict_users, dataset, num_users, opt):

    # 创建一个新的列表来存储修改后的数据集
    modified_dataset = []
    # 创建一个新的字典来存储修改后的用户
    modified_dict_users = {}

    # 减少最后两个用户的数据量到原来的50%
    if opt == 'less':

        # ll = len(dict_users[0]) * 0.2
        # for i in range(num_users - 2, num_users):
        #     while len(dict_users[i]) > ll:
        #         dict_users[i].pop()

        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                modified_dataset.append(dataset[idx])

        # 重新分配数据
        modified_num_items = int(len(dict_users[0]) * 0.5)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(
                np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    # 调整最后四个用户的数据量，最后两个用户到30%，前两个到50%
    # elif opt == 'less_rank':
    #     ll = len(dict_users[0]) * 0.50
    #     for i in range(num_users - 4, num_users - 2):
    #         while len(dict_users[i]) > ll:
    #             dict_users[i].pop()
    #     ll = len(dict_users[0]) * 0.30
    #     for i in range(num_users - 2, num_users):
    #         while len(dict_users[i]) > ll:
    #             dict_users[i].pop()

    # 对最后两个用户的数据应用随机噪声，并将修改后的数据加回到dataset中
    elif opt == 'random':
        def F(item):
            x, y = item
            # 生成一个与x形状相同的随机噪声数据。
            # 这里使用的是PyTorch库的torch.randn函数，它生成一个形状和x相同的张量，其元素从标准正态分布中随机采样。
            x = torch.randn(x.shape)
            # img = torchvision.transforms.ToPILImage()(x)
            # img.show()
            # img.save(f'./log/random_examples/{y}.bmp')
            item = x, y
            return item

        dataset.setup(F, opt)

        for i in range(num_users - 2, num_users):
            for idx in dict_users[i]:
                dataset.add_item(idx)

    # 对最后两个用户的数据应用随机噪声
    elif opt == 'noise':

        # 为图片添加椒盐噪声
        def add_salt_pepper_noise(item, salt_prob=0.01, pepper_prob=0.01):
            """
            为图片添加椒盐噪声
            :param item: 包含图像x和标签y的元组
            :param salt_prob: 盐（白点）噪声的概率
            :param pepper_prob: 椒（黑点）噪声的概率
            :return: 添加椒盐噪声后的图像和原始标签
            """
            x, y = item
            # 创建一个与原始图片相同大小的随机矩阵
            noise = torch.rand_like(x)

            # 添加盐噪声
            salt_mask = noise < salt_prob
            noisy_x = torch.where(salt_mask, torch.ones_like(x), x)

            # 添加椒噪声
            pepper_mask = noise < pepper_prob
            noisy_x = torch.where(pepper_mask, torch.zeros_like(x), noisy_x)

            return noisy_x, y

        # 高斯噪声添加函数 add_gaussian_noise
        def add_gaussian_noise(item, mean=0.0, std=1.0):
            x, y = item
            # 生成均值为mean，标准差为std的高斯噪声
            noise = torch.randn_like(x) * std + mean
            noisy_x = x + noise
            return noisy_x, y

        # 添加噪声到特定用户的数据（最后两个用户）
        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                # 高斯噪声
                noisy_item = add_gaussian_noise(original_item, mean=0.5, std=1.5)
                # 椒盐噪声
                # noisy_item = add_salt_pepper_noise(original_item, salt_prob=0.30, pepper_prob=0.30)
                modified_dataset.append(noisy_item)

        # 重新分配带噪声的数据
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    elif opt == 'gaussian_noise':
        def add_gaussian_noise(item, mean=0.0, std=1.0):
            x, y = item
            noise = torch.randn_like(x) * std + mean
            noisy_x = x + noise
            return noisy_x, y

        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                noisy_item = add_gaussian_noise(original_item, mean=0.5, std=1.5)
                modified_dataset.append(noisy_item)
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    elif opt == 'salt_pepper_noise':
        def add_salt_pepper_noise(item, salt_prob=0.01, pepper_prob=0.01):
            x, y = item
            noise = torch.rand_like(x)
            salt_mask = noise < salt_prob
            noisy_x = torch.where(salt_mask, torch.ones_like(x), x)
            pepper_mask = noise < pepper_prob
            noisy_x = torch.where(pepper_mask, torch.zeros_like(x), noisy_x)
            return noisy_x, y

        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                noisy_item = add_salt_pepper_noise(original_item, salt_prob=0.30, pepper_prob=0.30)
                modified_dataset.append(noisy_item)
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    # 随机更改最后两个用户数据的标签，引入标签错误
    elif opt == 'mislabel':

        def mislabel(item):
            x, y = item
            random_label = random.randint(0, 9)
            return x, random_label

        # 为最后两个用户的数据集中的每个数据项随机更改标签
        for user_id in [num_users - 2, num_users - 1]:
            for idx in dict_users[user_id]:
                original_item = dataset[idx]
                mislabeled_item = mislabel(original_item)
                modified_dataset.append(mislabeled_item)

        # 匹配最后两个用户的数据集序列
        modified_num_items = int(len(modified_dataset) / 2)
        modified_dict_users, modified_all_idxs = {}, [i for i in range(len(modified_dataset))]
        for user_id in [num_users - 2, num_users - 1]:
            modified_dict_users[user_id] = set(np.random.choice(modified_all_idxs, modified_num_items, replace=False))
            modified_all_idxs = list(set(modified_all_idxs) - modified_dict_users[user_id])

    # 对最后三四个用户的数据应用随机噪声，并且随机更改最后两个用户数据的标签
    elif opt == 'noise_mislabel':

        def add_gaussian_noise(item, mean=0.0, std=1.0):
            x, y = item
            # 生成均值为mean，标准差为std的高斯噪声
            noise = torch.randn_like(x) * std + mean
            noisy_x = x + noise
            return noisy_x, y

        def mislabel(item):
            x, y = item
            random_label = random.randint(0, 9)
            return x, random_label

        for user_id in [num_users-4, num_users-3, num_users-2, num_users-1]:  # 6 7 8 9
            if user_id < num_users-2:  # 6 7
                for idx in dict_users[user_id]:
                    original_item = dataset[idx]
                    noisy_item = add_gaussian_noise(original_item, mean=0.5, std=1.5)
                    modified_dataset.append(noisy_item)
            else:  # 8 9
                for idx in dict_users[user_id]:
                    original_item = dataset[idx]
                    mislabeled_item = mislabel(original_item)
                    modified_dataset.append(mislabeled_item)

        modified_num_items = int(len(modified_dataset) / 4)
        modified_noise_idxs = [i for i in range(2*modified_num_items)]
        modified_mislabel_idxs = [j+2*modified_num_items for j in modified_noise_idxs]

        for user_id in [num_users-4, num_users-3, num_users-2, num_users-1]:
            if user_id < num_users-2:  # 6-7
                modified_dict_users[user_id] = set(
                    np.random.choice(modified_noise_idxs, modified_num_items, replace=False))
                modified_noise_idxs = list(set(modified_noise_idxs) - modified_dict_users[user_id])
            else:  # 8-9
                modified_dict_users[user_id] = set(
                    np.random.choice(modified_mislabel_idxs, modified_num_items, replace=False))
                modified_mislabel_idxs = list(set(modified_mislabel_idxs) - modified_dict_users[user_id])

    # 不进行任何特别的数据处理
    elif opt == 'normal':
        pass
    else:
        print('No such option')
        exit(1)

    return modified_dict_users, modified_dataset


def mnist_iid(dataset, num_users, opt='normal'):
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

    # modified_dataset 列表用来来存储修改后的数据集
    # modified_dict_users 字典来记录修改后的用户和用户数据
    modified_dict_users, modified_dataset = specical_processing(dict_users, dataset, num_users, opt)

    return dict_users, modified_dict_users, modified_dataset


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users, opt='normal'):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    # dict_users: a dictionary to store users and their corresponding image indexes
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs,
                                             num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    modified_dict_users, modified_dataset = specical_processing(dict_users, dataset, num_users, opt)

    return dict_users, modified_dict_users, modified_dataset


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards 表示数据集被划分成的分片数量
    # num_imgs 表示每个分片包含的图像数量
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]

    # 始化了一个字典 dict_users，用于存储用户和其对应的图像索引。
    # 字典的键是用户 ID，值是一个空的 NumPy 数组，用于存储图像索引。
    dict_users = {i: np.array([]) for i in range(num_users)}

    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.train_labels)

    # sort labels
    # 将数据集中的图像按照标签值的升序进行排序
    '''
    将数据集中的图像按照标签值的升序进行排序，以确保后续的数据分配是按照标签进行的，从而实现非独立同分布的数据分配。
    idxs_labels = np.vstack((idxs, labels))：这行代码使用 NumPy 的 vstack 函数将图像索引数组 idxs 和标签数组 labels 垂直堆叠起来，形成一个包含两行的数组。
    第一行是图像索引，第二行是相应的标签。
    idxs_labels[:, idxs_labels[1, :].argsort()]：这行代码用于对标签进行排序。具体操作如下：
        idxs_labels[1, :] 选择了数组中的第二行，也就是标签。
        argsort() 函数对第二行的标签进行排序，并返回排序后的索引。排序后的索引会按照标签值的升序排列。
        最后，idxs_labels[:, ...] 使用排序后的索引来重新排列原始的两行数据，这样，图像索引和标签就按照标签值的升序排列了。
    idxs = idxs_labels[0, :]：最后，代码将排序后的图像索引提取出来，赋值给变量 idxs，以便后续使用。
    '''
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    # 为每个用户分配了不同的非独立同分布（Non-IID）数据子集。
    '''
    在每次迭代中，代码从 idx_shard 中随机选择两个分片的索引，并将它们存储在 rand_set 集合中。
    这里假设每个用户会随机选择两个分片，以获得非均匀分布的数据。
    将已经选择的分片从 idx_shard 中移除，以确保不重复选择相同的分片。
    将选定的分片中的图像索引添加到每个用户的 dict_users[i] 数组中，以创建非独立同分布的用户数据集。
    '''
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
