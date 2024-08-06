# -*- coding: utf-8 -*-
# Python version: 3.11
"""
Created on 12/12/2023

@author: junliu
"""

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNetCifar
from utils import mkdirs, get_dataset, average_weights, average_gradients
from utils import exp_details, gpu_is_available, show_training_state_diagram
from utils import calculate_contribution, calculate_similarity, calculate_final_contribution
from utils import normalize_global_list_values, show_final_contribution

import datetime
import json
import logging
import random

# import pdb


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check the gpu device
gpu_is_available()
# Prints the current working directory
print('Current working directory is:\n{}'.format(os.getcwd()))

if __name__ == '__main__':
    start_time = time.time()

    ### ## # CONFIGURING LOG FILES # ## ###

    args = args_parser()
    exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        argument_path = args.log_file_name + '.json'

    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    log_path = args.log_file_name + '.log'

    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info('device: %s', str(device))

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info('* Partitioning data *')

    # Load dataset and user groups
    train_dataset, test_dataset, user_groups, modified_user_groups, modified_dataset = get_dataset(args)
    logger.info('Data statistics: %s' % str(user_groups))

    # pdb.set_trace()

    ### ## # BUILD MODEL # ## ###

    logger.info('* Build model *')
    # Convolutional neural network
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'resnet':
        # 使用预定义的 resnet18 模型：
        global_model = ResNetCifar(args=args)

    # Multi-layer perceptron
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    logger.info('global_model:\n{}'.format(global_model))

    global_model.to(device)
    global_model.train()

    # Copy weights
    global_weights = global_model.state_dict()
    # Print the model weights
    # print('global_weights:\n{}\n'.format(global_weights))
    # logger.info('global_weights:\n{}'.format(global_weights))
    logger.info("#" * 100)

    ### ## # START TRAINING # ## ###

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []

    print_every = 1
    val_loss_pre, counter = 0, 0

    logger.info('* Start training *')

    ## # Define the contribution dictionary # ##

    # contri_dict: used to store the model contribution in each round of communication
    # key: comm_round
    # value: local model contribution dictionary
    contri_dict = {}
    # simi_global_list: list of the global model similarity
    simi_global_list = []
    # Create a list containing the indices of all parties
    party_list = [i for i in range(args.num_users)]
    # Store the index list of the parties in each round
    party_list_rounds = []

    for round in tqdm(range(args.comm_round)):
        local_weights, local_gradients, local_losses = [], [], []
        print(f'\n | Global Training Round : {round} |\n')
        logger.info(f'| Global Training Round : {round} |')

        global_model.train()

        # 预训练阶段所有客户端均参与
        if round < args.pre_round:
            # n_party_per_round clients are randomly selected to participate in federated training
            n_party_per_round = args.num_users
        # 非预训练阶段只有被选中的客户端参与
        else:
            n_party_per_round = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(args.num_users), n_party_per_round, replace=False)
        party_list_rounds.append(idxs_users)

        print('in comm round: {}, idxs_users: {}'.format(round, party_list_rounds[-1]))
        logger.info('in comm round: {}, idxs_users: {}'.format(round, party_list_rounds[-1]))

        # contri_dict_sub：current communication round model contribution subdictionary
        # key: comm_round
        # value: Local model contribution dictionary
        contri_dict_sub = {'comm_' + str(round): {'local_' + str(i): 0.0 for i in party_list}}
        contri_dict.update(contri_dict_sub)

        ## # LOCAL TRAINING # ##

        for idx in idxs_users:

            # 替换特殊用户的训练数据
            if args.opt != 'normal' and idx in [args.num_users - 2, args.num_users - 1]:
            # if args.opt != 'normal' and idx in [args.num_users-4, args.num_users-3, args.num_users-2, args.num_users-1]:
                print('specially processed data --> num_users: {}'.format(idx))
                logger.info('specially processed data --> num_users: {}'.format(idx))
                local_model = LocalUpdate(args=args,
                                          dataset=modified_dataset,
                                          idxs=modified_user_groups[idx],
                                          logger=logger)

            else:
                print('normally processed data --> num_users: {}'.format(idx))
                logger.info('normally processed data --> num_users: {}'.format(idx))
                local_model = LocalUpdate(args=args,
                                          dataset=train_dataset,
                                          idxs=user_groups[idx],
                                          logger=logger)
            # Performing local training
            w, g, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                    global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_gradients.append(copy.deepcopy(g))
            local_losses.append(copy.deepcopy(loss))

        # aggregate global weight
        global_weights = average_weights(local_weights)
        # print('global_weights: {}'.format(global_weights))

        # aggregate global gradient
        global_gradients = average_gradients(local_gradients)
        # print('global_gradients: {}'.format(global_gradients))

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        ## # Calculate the local contribution # ##

        if round < args.pre_round:
            print('pre_round: {}, now_round: {}'.format(args.pre_round, round+1))
            logger.info('pre_round: {}, now_round: {}'.format(args.pre_round, round+1))
            if round+1 == args.pre_round:
                # global_weights_base = global_weights
                global_weights_base = copy.deepcopy(global_weights)

        else:
            # *========== 修改 ========== 修改 ========== 修改 ========== #
            simi_global = calculate_similarity(global_weights_base, global_weights)
            simi_global_list.append(simi_global)
            print('simi_global_list: {}'.format(simi_global_list))
            logger.info('simi_global_list: {}'.format(simi_global_list))
            # 计算本地模型贡献，1.根据模型梯度；2.根据模型权重
            # 根据模型梯度
            if args.get_gradients:
                contri_list = calculate_contribution(global_gradients, local_gradients, idxs_users)
            # 根据模型权重（默认）
            else:
                contri_list = calculate_contribution(global_weights, local_weights, idxs_users)

            cnt_i = 0
            while cnt_i < len(contri_list):
                for item in idxs_users:
                    contri_dict['comm_' + str(round)]['local_' + str(item)] = contri_list[cnt_i]
                    cnt_i += 1

            print('\nin comm round: {}, idxs_users: {}'.format(round, idxs_users))
            print('The contribution of the federated user is:\n{}\n'.format(contri_dict['comm_' + str(round)]))

            logger.info('in comm round: {}, idxs_users: {}'.format(round, idxs_users))
            logger.info('The contribution of the federated user is: {}'.format(contri_dict['comm_' + str(round)]))
            # ========== 修改 ========== 修改 ========== 修改 ==========* #

            # ========== 修改的原型
            # # contri_list: local model contribution list (only participants in the current round)
            # contri_list = calculate_contribution(global_weights, local_weights, idxs_users)
            #
            # # simi_global: Global model similarity (warmed up baseline model vs global model per round)
            # simi_global = calculate_similarity(global_weights_base, global_weights)
            # simi_global_list.append(simi_global)
            # print('simi_global_list: {}'.format(simi_global_list))
            # logger.info('simi_global_list: {}'.format(simi_global_list))
            #
            # # The contribution values are added to the contribution dictionary in the order of participation in the training
            # cnt_i = 0
            # while cnt_i < len(contri_list):
            #     for item in idxs_users:
            #         contri_dict['comm_' + str(round)]['local_' + str(item)] = contri_list[cnt_i]
            #         cnt_i += 1
            #
            # print('\nin comm round: {}, idxs_users: {}'.format(round, idxs_users))
            # print('The contribution of the federated user is:\n{}\n'.format(contri_dict['comm_' + str(round)]))
            #
            # logger.info('in comm round: {}, idxs_users: {}'.format(round, idxs_users))
            # logger.info('The contribution of the federated user is: {}'.format(contri_dict['comm_' + str(round)]))
            # ========== 修改的原型

        ## # Compute the local client precision # ##

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        for c in range(args.num_users):
            local_model = LocalUpdate(args=args,
                                      dataset=train_dataset,
                                      idxs=user_groups[c],
                                      logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (round+1) % print_every == 0:
            print(f'\nAvg Training Stats after {round+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        logger.info(f'Avg Training Stats after {round+1} global rounds:')
        logger.info(f'>> Training Loss : {np.mean(np.array(train_loss))}')
        logger.info('>> Train Accuracy: {:.2f}% '.format(100*train_accuracy[-1]))

    logger.info('* End of Training *')
    logger.info("#" * 100)


    ### ## # TEST INFERENCE # ## ###


    logger.info('* Test Inference *')
    # Normalize the global model similarity list
    # contri_global_list = normalize_list_values(simi_global_list)
    # 此处可以考虑重新设计normalize_list_values函数，将全局模型的相似度列表映射到高维
    # [0,1]-->[0,100]
    contri_global_list = normalize_global_list_values(simi_global_list, 100)

    # Calculating final contribution after completion of training
    contri_final = calculate_final_contribution(contri_dict, contri_global_list, args.pre_round)
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f'\nResults after {args.comm_round} global rounds of training:')
    print('|---- Avg Train Accuracy: {:.2f}%'.format(100*train_accuracy[-1]))
    print('|---- Test Accuracy: {:.2f}%'.format(100*test_acc))

    logger.info(f'Results after {args.comm_round} global rounds of training:')
    logger.info('|---- Avg Train Accuracy: {:.2f}%'.format(100*train_accuracy[-1]))
    logger.info('|---- Test Accuracy: {:.2f}%'.format(100*test_acc))

    # The contribution of all users is counted
    print('contri_global_list: {}'.format(contri_global_list))
    print('Participants in each round of federated training are: \n{}'.format(party_list_rounds))
    # print('The total contribution of federated users is: \n{}'.format(contri_dict))
    print('The final contribution of federated users is: \n{}'.format(contri_final))

    # logger.info('contri_global_list: {}'.format(contri_global_list))
    logger.info('Participants in each round of federated training are: \n{}'.format(party_list_rounds))
    logger.info('The total contribution of federated users is: \n{}'.format(contri_dict))
    logger.info('The final contribution of federated users is: \n{}'.format(contri_final))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_P[{}]_O[{}].pkl'.format(args.dataset,
                                                                                           args.model,
                                                                                           args.comm_round,
                                                                                           args.frac,
                                                                                           args.iid,
                                                                                           args.local_ep,
                                                                                           args.local_bs,
                                                                                           args.pre_round,
                                                                                           args.opt)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    # Output training state graph (training loss, training accuracy)
    show_training_state_diagram(args, train_loss, train_accuracy)
    show_final_contribution(args, contri_final)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    logger.info('Total Run Time: {0:0.4f}'.format(time.time()-start_time))
