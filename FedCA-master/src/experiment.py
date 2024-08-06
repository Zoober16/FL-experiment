# -*- coding: utf-8 -*-
# Python version: 3.11
"""
Created on 12/12/2023

@author: junliu
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Arguments():
    def __init__(self):
        self.dataset = 'mnist'  # mnist, fmnist, cifar
        self.model = 'cnn'
        self.comm_round = 20
        self.frac = 1.0
        self.iid = 1
        self.pre_round = 2
        self.opt1 = 'normal'  # normal, noise, mislabel
        self.opt2 = 'mislabel'
        self.path = ''


def show_final_contribution(args, contri_final1, contri_final2):

    matplotlib.use('Agg')

    users_list = list(contri_final1.keys())
    x_arange = np.arange(len(users_list))
    contri_list1 = list(contri_final1.values())
    contri_list2 = list(contri_final2.values())

    plt.figure(figsize=(20, 10), dpi=200)
    plt.grid(axis="y", c='#d2c9eb', linestyle='--', zorder=0)

    bar_width = 0.4

    plt.bar(x_arange - bar_width / 2, contri_list1, label=args.opt1, color='#ff9999', width=bar_width,
            edgecolor='black', linewidth=2.0, zorder=10)
    plt.bar(x_arange + bar_width / 2, contri_list2, label=args.opt2, color='#9ed9d5', width=bar_width,
            edgecolor='black', linewidth=2.0, zorder=10)

    for x, ii, jj in zip(x_arange, contri_list1, contri_list2):
        plt.text(x - bar_width / 2, ii + 0.002, '%.4f' % ii, ha='center', fontproperties='Times New Roman',
                 fontsize=14, zorder=10)
        plt.text(x + bar_width / 2, jj + 0.002, '%.4f' % jj, ha='center', fontproperties='Times New Roman',
                 fontsize=14, zorder=10)

    plt.ylim(0, 0.2)

    plt.title('Contributions Of Federated Users', fontproperties='Times New Roman', fontsize=40)
    plt.xlabel('Local Users', fontproperties='Times New Roman', fontsize=40)
    plt.ylabel('Contributions', fontproperties='Times New Roman', fontsize=40)

    plt.xticks(x_arange, labels=users_list, fontproperties='Times New Roman', fontsize=20)
    plt.yticks(fontproperties='Times New Roman', fontsize=20)

    plt.legend(prop={'family': 'Times New Roman', 'size': 25}, ncol=2)

    # plt.show()
    plt.savefig('{}/fed_{}_{}_{}_C[{}]_iid[{}]_P[{}]_O[{}-{}]_contri.png'.
                format(args.path,
                       args.dataset,
                       args.model,
                       args.comm_round,
                       args.frac,
                       args.iid,
                       args.pre_round,
                       args.opt1,
                       args.opt2))

    return


if __name__ == '__main__':

    FL_params = Arguments()

    contri_normal = {'local_0': 0.09998324613081136, 'local_1': 0.10000038670019305, 'local_2': 0.10001980607406386, 'local_3': 0.10001010846987007, 'local_4': 0.10000481793162534, 'local_5': 0.10002044907420805, 'local_6': 0.09998278302142123, 'local_7': 0.09999285722724863, 'local_8': 0.09998335055704125, 'local_9': 0.1000021948135169}
    contri_noise = {'local_0': 0.10326079597202048, 'local_1': 0.10411914327126454, 'local_2': 0.10377512558271929, 'local_3': 0.10332411552582221, 'local_4': 0.10333570744729878, 'local_5': 0.10363202489691746, 'local_6': 0.10367887433067138, 'local_7': 0.10250194825376688, 'local_8': 0.08723581295797307, 'local_9': 0.0851364517615459}
    contri_mislabel = {'local_0': 0.11422582781415765, 'local_1': 0.1148034810693355, 'local_2': 0.11457921539503926, 'local_3': 0.11452602135816634, 'local_4': 0.11422867171295098, 'local_5': 0.11424387583633294, 'local_6': 0.11451018080834043, 'local_7': 0.11428054753442497, 'local_8': 0.0311000357092731, 'local_9': 0.05350214276197884}

    show_final_contribution(FL_params, contri_normal, contri_mislabel)
