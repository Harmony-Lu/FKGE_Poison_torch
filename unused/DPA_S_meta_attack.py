#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 17:14
# @Author  : zhixiuma
# @File    : DPA_S.py
# @Project : FKGEAttack
# @Software: PyCharm

import numpy as np
import random
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict as ddict
# from dataloader import *
import os
import copy
import logging
from kge_model import KGEModel
from torch import optim
import torch.nn.functional as F
import itertools
from itertools import permutations
from random import choice
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from process_data.generate_client_data_2 import TrainDataset, generate_new_id

# from metattack import MetaApprox
from unused.metattack import Metattack
import seaborn as sns
from matplotlib import pyplot as plt

# import utils
# import tensorflow as tf
# import numpy as np
# import scipy.sparse as sp
# import  tf_slim as slim
# try:
#     from tqdm import tqdm
# except ImportError:
#     tqdm = lambda x, desc=None: x

# class KGE_Attack:
#     '''
#         Base class for attacks on GNNs.
#     '''
#     def __init__(self, args, poisoned_triples, rel_embed, victim_client_ent_embed, poisoned_ent_embed, victim_client=0) :
#         self.args = args
#         self.kge_model = KGEModel(args, args.server_model)
#         self.poisoned_triples = poisoned_triples
        
        # train_dataset = TrainDataset(self.poisoned_triples, self.args.nentity, self.args.num_neg)
        # self.train_dataloader = DataLoader(
        #     train_dataset,
        #     batch_size=self.args.batch_size,
        #     shuffle=True,
        #     collate_fn=TrainDataset.collate_fn
        # )
        
#         self.nentity = args.nentity
#         self.nrelation = args.nrelation

#         triplet_adjacency_matrix = self.triplet2adjacency(self.poisoned_triples) # 二维Array
#         self.triplet_matrix = triplet_adjacency_matrix.requires_grad_(True)

#         pass
            
#     def filter_potential_singletons(self):
#         """
#         Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
#         the entry have degree 1 and there is an edge between the two nodes.

#         Returns
#         -------
#         tf.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
#         where the returned tensor has value 0.

#         """
#         degrees = tf.reduce_sum(self.modified_adjacency, axis=0)
#         degree_one = tf.equal(degrees, 1, name="degree_equals_one")
#         resh = tf.reshape(tf.tile(degree_one, [self.N]), [self.N, self.N], name="degree_one_square")
#         l_and = tf.logical_and(resh, tf.equal(self.modified_adjacency, 1))
#         logical_and_symmetric = tf.logical_or(l_and, tf.transpose(l_and))
#         flat_mask = tf.cast(tf.logical_not(tf.reshape(logical_and_symmetric, [-1])), self.dtype)
#         return flat_mask
#         # flat_mask = np.full((self.nentity, self.nentity), 1.0)
#         # return flat_mask


# class KGE_MetaApprox(KGE_Attack):
#     """
#         Class for attacking FKGE with meta gradients.
#     """
#     def __init__(self, args, poisoned_triples, rel_embed, victim_client_ent_embed, poisoned_ent_embed, victim_client=0):
#         super().__init__(args, poisoned_triples, rel_embed, victim_client_ent_embed, poisoned_ent_embed, victim_client)
#         pass 
    
#     def build(self):
#         print('KGE_MetaApprox.build()')
#         pass


# class KGE_Meta(KGE_Attack):
    # """
    #     Class for attacking FKGE with meta gradients.
    # """
    # def __init__(self, args, poisoned_triples, rel_embed, victim_client_ent_embed, poisoned_ent_embed, victim_client=0):
    #     super().__init__(args, poisoned_triples, rel_embed, victim_client_ent_embed, poisoned_ent_embed, victim_client)
        
    #     self.total_loss = None
    #     self.optimizer = None
    #     self.train_op = None
    #     self.ll_ratio = None
        
    # def triplet2adjacency(self, triplet):
    #         # 初始化为0
    #         # triplet_adjacency_matrix = np.full((self.nrelation, self.nentity, self.nentity), 0)
    #         triplet_adjacency_matrix = np.zeros((self.nentity, self.nentity))
    #         for t in self.poisoned_triples:
    #             h = t[0]
    #             r = t[1]
    #             t = t[2]
    #             # triplet_adjacency_matrix[h][t] = r+1 # 矩阵的取值：1-237    0：两个entity之间没有任何关系 
    #             triplet_adjacency_matrix[h][t] = 1 # 矩阵的取值：1/0    0：两个entity之间没有任何关系 
    #         return triplet_adjacency_matrix # 二维Array
    
    # def build(self):
    #     print('KGE_Meta.build()')
    #     optimizer = optim.Adam([{'params': self.rel_embed}, # 刚初始化的rel_embed
    #                             # {'params': self.triplet_adjacency_matrix}, # victim client ent_embed
    #                             {'params': self.victim_client_ent_embed}, # victim client ent_embed
    #                             {'params': self.poisoned_ent_embed}], lr=float(self.args.lr)) # 刚初始化的ent_embed
    #     losses = []
    #     head = np.array(self.poisoned_tri)[:, 0] # 制造的有毒三元组的头尾实体
    #     tail = np.array(self.poisoned_tri)[:, 2]
        # for i in range(int(self.args.local_epoch)):
        #     # the training dataset(D_p = T_1 + t_p) + shadow model
        #     print('************** server training loss ******************')
        #     for batch in self.train_dataloader:
        #         positive_sample, negative_sample, sample_idx = batch

        #         positive_sample = positive_sample.to(self.args.gpu)
        #         negative_sample = negative_sample.to(self.args.gpu)

        #         negative_score = self.kge_model((positive_sample, negative_sample),
        #                                         self.rel_embed, self.poisoned_ent_embed)

        #         negative_score = (F.softmax(negative_score * float(self.args.adversarial_temperature), dim=1).detach()
        #                           * F.logsigmoid(-negative_score)).sum(dim=1)

        #         positive_score = self.kge_model(positive_sample,
        #                                         self.rel_embed, self.poisoned_ent_embed, neg=False)

        #         positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        #         positive_sample_loss = - positive_score.mean()
        #         negative_sample_loss = - negative_score.mean()
        #         loss = (positive_sample_loss + negative_sample_loss) / 2

        #         victim_client_head_emb = self.victim_client_ent_embed[head].unsqueeze(dim=1) # victim client embedding
        #         victim_client_tail_emb = self.victim_client_ent_embed[tail].unsqueeze(dim=1)
        #         server_head_emb = self.poisoned_ent_embed[head].unsqueeze(dim=1)# shadow model trained embedding
        #         server_tail_emb = self.poisoned_ent_embed[tail].unsqueeze(dim=1)

        #         # minimize the embedding inconsistencies between the server and the victim client
        #         loss_dis = ((victim_client_head_emb - server_head_emb) + 
        #                     (victim_client_tail_emb - server_tail_emb)).mean()
                
        #         # loss = loss + loss_dis
        #         # loss = loss +0.000001*loss_dis
        #         loss = loss

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         losses.append(loss.item())

    #     return np.mean(losses)


    # def make_loss(self, ll_constraint=True, ll_cutoff=0.004):
    #     with self.graph.as_default():
    #         # This variable "stores" the gradients of every inner training step.
    #         # self.grad_sum = tf.Variable(np.zeros(self.nrelation*self.nentity*self.nentity), dtype=self.dtype) # 三维
    #         self.grad_sum = tf.Variable(np.zeros(self.nentity*self.nentity), dtype=self.dtype)   # 二维

    #         # self.ent_grad = tf.gradients(self.total_loss, self.triplet_changes)[0]
    #         self.ent_grad = tf.multiply(tf.gradients(self.total_loss, self.triplet_changes)[0],
    #                                           tf.reshape(self.modified_triplet, [-1])*-2 + 1, # meta_grad = gradient*(1-2*A)
    #                                           name="Adj_gradient")

    #         # Add the current gradient to the sum.
    #         self.grad_sum_add = tf.assign_add(self.grad_sum, self.ent_grad)

    #         # Make sure that the minimum entry is 0.
    #         self.grad_sum_mod = self.grad_sum - tf.reduce_min(self.grad_sum)

    #         # Set entries to 0 that could lead to singleton entities.
    #         singleton_mask = self.filter_potential_singletons()  # 暂时不考虑singleton_entity
    #         self.grad_sum_mod = tf.multiply(self.grad_sum_mod, singleton_mask)

    #         # Get argmax of the approximate meta gradients.
    #         adj_meta_approx_argmax = tf.argmax(self.grad_sum_mod)

    #         # Compute the index corresponding to the reverse direction of the edge (i.e. in the other triangle of the matrix).
    #         adj_argmax_transpose_ix = utils.ravel_index(utils.unravel_index_tf(adj_meta_approx_argmax,
    #                                                     [self.nentity, self.nentity])[::-1], [self.nentity, self.nentity])
            
    #         # Stack both indices to make sure our matrix remains symmetric.
    #         adj_argmax_combined = tf.stack([adj_meta_approx_argmax, adj_argmax_transpose_ix],
    #                                        name="Meta_approx_argmax_combined")
            
    #         # Add the change to the perturbations.
    #         self.triplet_update = tf.scatter_add(self.triplet_changes,
    #                                                indices=adj_argmax_combined,
    #                                                updates=-2 * tf.gather( tf.reshape(self.modified_adjacency, [-1]),
    #                                                         adj_argmax_combined) + 1)
            

    # def attack(self, perturbations, initialize=True):
    #     with self.graph.as_default():
    #         if initialize:
    #             self.session.run(tf.global_variables_initializer())

            
    #         opt_vars = [v for v in self.optimizer.variables()]
    #         for _it in tqdm(range(perturbations), desc="Perturbing graph"):
    #             self.session.run(self.grad_sum.initializer)
    #             self.session.run( [ v.initializer for v in opt_vars])
    #             for train_iter in range(self.train_iters):
    #                 self.session.run([self.train_op, self.grad_sum_add])
    #                 triplet_update = self.sesssion.run(self.triplet_update)
    #                 print("triplet_update: ", triplet_update)
    #             # 根据modified_triplet 计算一次中毒攻击后的 poisoned_triples，再得到当前的train_dataloader
    #             # # 
                
    #             # train_dataset = TrainDataset(self.poisoned_triples, self.args.nentity, self.args.num_neg)
    #             # self.train_dataloader = DataLoader(
    #             #     train_dataset,
    #             #     batch_size=self.args.batch_size,
    #             #     shuffle=True,
    #             #     collate_fn=TrainDataset.collate_fn
    #             # )


    #     pass

class Server(object):
    def __init__(self, args, nentity): # 初始化 self.ent_embed
        self.args = args
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / int(args.hidden_dim)])

        # the server initializes a global entity embeddings matrix randomly
        if args.client_model in ['RotatE', 'ComplEx']: # Server端聚合后的ent_embed
            self.ent_embed = torch.zeros(nentity, int(args.hidden_dim) * 2).to(args.gpu).requires_grad_()
            # self.ent_embed = torch.zeros(nentity, int(args.hidden_dim) * 2).to(args.gpu)
        else:
            self.ent_embed = torch.zeros(nentity, int(args.hidden_dim)).to(args.gpu).requires_grad_()
            # self.ent_embed = torch.zeros(nentity, int(args.hidden_dim)).to(args.gpu)
        nn.init.uniform_(
            tensor=self.ent_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        self.nentity = nentity

        self.train_dataloader = None
        self.victim_client = None
        # self.kge_model = KGEModel(args, args.server_model)

    # the TransE model used to perform inference attacks,
    def TransE(self, head, tail):
        r = head - tail
        return r

    def send_emb(self):
        return copy.deepcopy(self.ent_embed)

    def poison_attack_random(self,  victim_client=0): # 获取数据集 self.poisoned_triples
        self.victim_client=victim_client
        # Step1: Relation Inference
        # 0.Suppose the server knows the victim client's training triplet
        with open('process_data/client_data/' + self.args.dataset_name + '_' + str(self.args.num_client) + '_with_new_id.json', 'r') as file1:
            real_triples = json.load(file1)

        # head_list = (np.array(real_triples[victim_client]['train'])[:,[0]]).squeeze().tolist()
        # relation_list = (np.array(real_triples[victim_client]['train'])[:, [1]]).squeeze().tolist()
        # tail_list =(np.array(real_triples[victim_client]['train'])[:, [2]]).squeeze().tolist()

        if int(self.args.attack_entity_ratio)==0:
            self.poisoned_triples = real_triples[victim_client]['train']
            print('len(self.poisoned_triples:',len(self.poisoned_triples))
            return

        poisoned_triples = []
        self.poisoned_triples = poisoned_triples + real_triples[victim_client]['train']
        print("数据集：", type(self.poisoned_triples), len(self.poisoned_triples))


    def create_poison_dataset(self): # 初始化 self.rel_embed
        train_dataset = TrainDataset(self.poisoned_triples, self.args.nentity, self.args.num_neg)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )

        embedding_range = torch.Tensor([(self.args.gamma + self.args.epsilon) / int(self.args.hidden_dim)])
        if self.args.server_model in ['ComplEx']:
            self.rel_embed = torch.zeros(self.args.nrelation, int(self.args.hidden_dim) * 2).to(
                self.args.gpu).requires_grad_()
            # self.rel_embed = torch.zeros(self.args.nrelation, int(self.args.hidden_dim)*2).to(self.args.gpu)
        else:
            self.rel_embed = torch.zeros(self.args.nrelation, int(self.args.hidden_dim)).to(self.args.gpu).requires_grad_()
            # self.rel_embed = torch.zeros(self.args.nrelation, int(self.args.hidden_dim)).to(self.args.gpu)

        nn.init.uniform_(
            tensor=self.rel_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )

        self.ent_freq = torch.zeros(self.args.nentity)
        for e in np.array(self.poisoned_triples)[:, [0, 2]].reshape(-1):
            self.ent_freq[e] += 1
        self.ent_freq = self.ent_freq.unsqueeze(dim=0).to(self.args.gpu)  #

    # The server performs the poison aggregation
    def poison_aggregation(self, clients, ent_update_weights):

        ent_update_weights = torch.cat((ent_update_weights, self.ent_freq), dim=0)

        agg_ent_mask = ent_update_weights
        agg_ent_mask[ent_update_weights != 0] = 1

        ent_w_sum = torch.sum(agg_ent_mask, dim=0)
        ent_w = agg_ent_mask / ent_w_sum
        ent_w[torch.isnan(ent_w)] = 0

        if self.args.server_model in ['RotatE', 'ComplEx']:
            update_ent_embed = torch.zeros(self.nentity, int(self.args.hidden_dim) * 2).to(self.args.gpu)
        else:
            update_ent_embed = torch.zeros(self.nentity, int(self.args.hidden_dim)).to(self.args.gpu)

        for i, client in enumerate(clients):
            local_ent_embed = client.ent_embed.clone().detach()
            update_ent_embed += local_ent_embed * ent_w[i].reshape(-1, 1)
        # the server add the malicious entity embeddings (self.poisoned_ent_embed) to aggregation results (update_ent_embed).
        update_ent_embed += self.poisoned_ent_embed.clone().detach() * ent_w[-1].reshape(-1, 1)
        self.ent_embed = update_ent_embed.requires_grad_()

    # The server trains its shadow model to maximize the probability of the poisoned triplets.
    def train_poison_model(self, clients):
        if self.victim_client == None:
            self.victim_client = 0
        else:
            self.victim_client = self.victim_client

        self.victim_client_ent_embed = clients[self.victim_client].ent_embed

        if self.train_dataloader == None: # 初始化self.poisoned_triples：生成poisoned_triples
            self.poison_attack_random(victim_client=self.victim_client)

        self.create_poison_dataset() # 初始化self.rel_embed
        client_loss = self.server_poison_dynamic_update() # training of shadow model
        
    def test(self, triplet):
        print("...... test ......")
    # The Dynamic Poisoning Attack
    def server_poison_dynamic_update(self):
        self.poisoned_ent_embed = self.send_emb()
        perturbations = 50
        ori_poisoned_triples = self.poisoned_triples.copy()
        model = Metattack(self.args, self.poisoned_triples, self.rel_embed,
                          self.poisoned_ent_embed, self.victim_client_ent_embed)
        modified_poisoned_triples = model(ori_poisoned_triples, perturbations)

        runs = 10
        clean_acc = []
        attacked_acc = []
        print('=== testing kge_model on original(clean) graph ===')
        for i in range(runs):
            clean_acc.append(self.test(ori_poisoned_triples))
        print('=== testing kge_model on attacked graph ===')
        for i in range(runs):
            attacked_acc.append(self.test(modified_poisoned_triples))
        plt.figure(figsize=(6,6))
        sns.boxplot(x=["Acc. Clean", "Acc. Perturbed"], y=[clean_acc, attacked_acc])
        plt.title("Accuracy before/after {} perturbations using model {}".format(perturbations, self.args.server_model))
        plt.savefig("results.png", dpi=600)
        plt.show()
        # # The attack variants from the paper
        # variants = ["Meta-Train", "Meta-Self","A-Meta-Train", "A-Meta-Self", "A-Meta-Both"]
        # # Choose the variant you would like to try
        # # variant = "Meta-Self"
        # variant = "Meta-Train"
        # # variant = "A-Meta-Train"
        # assert variant in variants
        # enforce_ll_constrant = False
        # approximate_meta_gradient = False
        # if variant.startswith("A-"): # approximate meta gradient
        #     approximate_meta_gradient = True
        #     if "Train" in variant:
        #         lambda_ = 1 # meta-train和self-train各占的比例
        #     elif "Self" in variant:
        #         lambda_ = 0
        #     else:
        #         lambda_ = 0.5
        # if approximate_meta_gradient:
        #     print('KGE_MetaApprox')
        #     gcn_attack = KGE_MetaApprox(train_iters=train_iters, dtype=dtype)
        # else:
        #     print('KGE_Meta')
        #     gcn_attack = KGE_Meta(self.args, self.poisoned_triples, self.rel_embed, self.victim_client_ent_embed, self.poisoned_ent_embed, self.victim_client)
        #     print('KGE_Meta Over')
        # gcn_attack.build()
        # gcn_attack.make_loss(ll_constraint=enforce_ll_constrant)
        # gcn_attack.attack()
        # loss = gcn_attack.loss.eval(session=gcn_attack.session)
        # return loss

        # # self.rel_embed
        # # self.victim_client_ent_embed
        # # self.poisoned_ent_embed
        # # ------------------------------------------------------------------------------
        # optimizer = optim.Adam([{'params': self.rel_embed}, # 刚初始化的rel_embed
        #                         {'params': self.victim_client_ent_embed}, # victim client ent_embed
        #                         {'params': self.poisoned_ent_embed}], lr=float(self.args.lr)) # 刚初始化的ent_embed
        # losses = []
        # head = np.array(self.poisoned_tri)[:, 0]
        # tail = np.array(self.poisoned_tri)[:, 2]
        # for i in range(int(self.args.local_epoch)):
        #     # the training dataset(D_p = T_1 + t_p) + shadow model
        #     print('************** server training loss *********************')
        #     for batch in self.train_dataloader:
        #         positive_sample, negative_sample, sample_idx = batch

        #         positive_sample = positive_sample.to(self.args.gpu)
        #         negative_sample = negative_sample.to(self.args.gpu)

        #         negative_score = self.kge_model((positive_sample, negative_sample),
        #                                         self.rel_embed, self.poisoned_ent_embed)

        #         negative_score = (F.softmax(negative_score * float(self.args.adversarial_temperature), dim=1).detach()
        #                           * F.logsigmoid(-negative_score)).sum(dim=1)

        #         positive_score = self.kge_model(positive_sample,
        #                                         self.rel_embed, self.poisoned_ent_embed, neg=False)

        #         positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        #         positive_sample_loss = - positive_score.mean()
        #         negative_sample_loss = - negative_score.mean()
        #         loss = (positive_sample_loss + negative_sample_loss) / 2

        #         victim_client_head_emb = self.victim_client_ent_embed[head].unsqueeze(dim=1) # victim client embedding
        #         victim_client_tail_emb = self.victim_client_ent_embed[tail].unsqueeze(dim=1)
        #         server_head_emb = self.poisoned_ent_embed[head].unsqueeze(dim=1)# shadow model trained embedding
        #         server_tail_emb = self.poisoned_ent_embed[tail].unsqueeze(dim=1)

        #         # minimize the embedding inconsistencies between the server and the victim client
        #         loss_dis = ((victim_client_head_emb - server_head_emb) + (
        #                     victim_client_tail_emb - server_tail_emb)).mean()
                
        #         # loss = loss + loss_dis
        #         # loss = loss +0.000001*loss_dis
        #         loss = loss

        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #         losses.append(loss.item())
        #     # total_loss = np.sum(losses)
        #     # total_loss

        # return np.mean(losses)
        
    def poisoned_labels(self,poisoned_tri):
        labels = []
        mask = np.unique(poisoned_tri[:,[0,2]].reshape(-1))
        for i in range(len(poisoned_tri)):
            y = np.zeros([self.ent_embed.shape[0]], dtype=np.float32)
            y[mask] = 1
            labels.append(y)
        return labels

    def server_poisoned_eval(self,  poisoned_tri=None):

        results = ddict(float)

        if poisoned_tri != None:

            poisoned_tri = np.array(poisoned_tri).astype(int)
            head_idx, rel_idx, tail_idx = poisoned_tri[:, 0], poisoned_tri[:, 1], poisoned_tri[:, 2]
            labels = self.poisoned_labels(poisoned_tri)
            pred = self.kge_model((torch.IntTensor(poisoned_tri.astype(int)).to(self.args.gpu), None),
                                  self.rel_embed, self.ent_embed)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(torch.FloatTensor(labels).byte().to(self.args.gpu), -torch.ones_like(pred) * 10000000,
                               pred)
            pred[b_range, tail_idx] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]
            ranks = ranks.float()
            count = torch.numel(ranks)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item() / len(poisoned_tri)
            results['mrr'] += torch.sum(1.0 / ranks).item() / len(poisoned_tri)

            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k]) / len(poisoned_tri)  #
            return results


class Client(object):
    def __init__(self, args, client_id, data, train_dataloader,
                 valid_dataloader, test_dataloader, rel_embed,all_ent):
        self.args = args
        self.data = data
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.rel_embed = rel_embed
        self.client_id = client_id
        self.all_ent = all_ent

        self.score_local = []
        self.score_global = []

        self.kge_model = KGEModel(args, args.client_model)
        self.ent_embed = None

    def __len__(self):
        return len(self.train_dataloader.dataset)

    def client_update(self):
        optimizer = optim.Adam([{'params': self.rel_embed},
                                {'params': self.ent_embed}], lr=float(self.args.lr))
        losses = []
        for i in range(int(self.args.local_epoch)):
            for batch in self.train_dataloader:
                positive_sample, negative_sample, sample_idx = batch

                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)
                negative_score = self.kge_model((positive_sample, negative_sample),
                                                self.rel_embed, self.ent_embed)

                negative_score = (F.softmax(negative_score * float(self.args.adversarial_temperature), dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

                positive_score = self.kge_model(positive_sample,
                                                self.rel_embed, self.ent_embed, neg=False)

                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()

                loss = (positive_sample_loss + negative_sample_loss) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

        return np.mean(losses)

    def poisoned_labels(self,poisoned_tri):
        labels = []
        mask = np.unique(poisoned_tri[:,[0,2]].reshape(-1))
        for i in range(len(poisoned_tri)):
            y = np.zeros([self.ent_embed.shape[0]], dtype=np.float32)
            y[mask] = 1
            labels.append(y)
        return labels

    def client_eval(self, istest=False,poisoned_tri=None):
        if istest:
            dataloader = self.test_dataloader
        else:
            dataloader = self.valid_dataloader

        results = ddict(float)

        if poisoned_tri!=None:

            poisoned_tri = np.array(poisoned_tri).astype(int)
            head_idx, rel_idx, tail_idx = poisoned_tri[:, 0],poisoned_tri[:, 1], poisoned_tri[:, 2]
            labels = self.poisoned_labels(poisoned_tri)
            pred = self.kge_model((torch.IntTensor(poisoned_tri.astype(int)).to(self.args.gpu), None),
                                  self.rel_embed, self.ent_embed)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(torch.FloatTensor(labels).byte().to(self.args.gpu), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]
            ranks = ranks.float()
            count = torch.numel(ranks)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item() /len(poisoned_tri)
            results['mrr'] += torch.sum(1.0 / ranks).item()/len(poisoned_tri)

            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])/len(poisoned_tri)
            return results

        for batch in dataloader:
            triplets, labels = batch
            triplets, labels = triplets.to(self.args.gpu), labels.to(self.args.gpu)
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            pred = self.kge_model((triplets, None),
                                  self.rel_embed, self.ent_embed)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]

            ranks = ranks.float()
            count = torch.numel(ranks)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in results.items():
            if k != 'count':
                results[k] /= results['count']

        return results

from process_data.generate_client_data_2 import read_triples

class DPA_S(object):
    def __init__(self, args, all_data):
        self.args = args

        # assign triplets to each client
        train_dataloader_list, valid_dataloader_list, test_dataloader_list, \
        self.ent_freq_mat, rel_embed_list, nentity, nrelation, all_ent_list = read_triples(all_data, args)

        self.args.nentity = nentity
        self.args.nrelation = nrelation

        # clients
        self.num_clients = len(train_dataloader_list)
        self.clients = [
            Client(args, i, all_data[i], train_dataloader_list[i], valid_dataloader_list[i],
                   test_dataloader_list[i], rel_embed_list[i],all_ent_list[i]) for i in range(self.num_clients)
        ]

        self.server = Server(args, nentity)

        self.total_test_data_size = sum([len(client.test_dataloader.dataset) for client in self.clients])
        self.test_eval_weights = [len(client.test_dataloader.dataset) / self.total_test_data_size for client in
                                  self.clients]

        self.total_valid_data_size = sum([len(client.valid_dataloader.dataset) for client in self.clients])
        self.valid_eval_weights = [len(client.valid_dataloader.dataset) / self.total_valid_data_size for client in
                                   self.clients]

    def write_training_loss(self, loss, e):
        self.args.writer.add_scalar("training/loss", loss, e)

    def write_evaluation_result(self, results, e):
        self.args.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.args.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.args.writer.add_scalar("evaluation/hits5", results['hits@5'], e)
        self.args.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def save_checkpoint(self, e):
        state = {'ent_embed': self.server.ent_embed,
                 'server_rel_embed':self.server.rel_embed,
                 'posioned_tri':self.server.poisoned_tri,
                 'victim_client':self.server.victim_client,
                 'rel_embed': [client.rel_embed for client in self.clients]}

        # delete previous checkpoint
        for filename in os.listdir(self.args.state_dir):
            if self.args.name in filename.split('-') and os.path.isfile(os.path.join(self.args.state_dir, filename)):
                os.remove(os.path.join(self.args.state_dir, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name+ '-' + str(e) + '.ckpt'))

    def save_model(self, best_epoch):
        os.rename(os.path.join(self.args.state_dir, self.args.name+'-' + str(best_epoch) + '.ckpt'),os.path.join(self.args.state_dir, self.args.name+ '.best'))

    def send_emb(self):
        for k, client in enumerate(self.clients):
            client.ent_embed = self.server.send_emb()

    def server_dynamic_attack(self,clients):
        self.server.train_poison_model(clients)

    def train(self):

        n_sample = max(round(self.args.fraction * self.num_clients), 1)
        sample_set = np.random.choice(self.num_clients, n_sample, replace=False)

        best_epoch = 0
        best_mrr = 0
        bad_count = 0

        for num_round in range(self.args.max_round):
            # the server sends the global entity embeddings matrix to all clients
            self.send_emb()
            # Local Client Model Training
            round_loss = 0
            for k in iter(sample_set):
                # client_loss = self.clients[k].client_update()
                client_loss = 0
                round_loss += client_loss
            round_loss /= n_sample

            # Step3: Shadow Model Training
            # The server first trains a shadow model to perform poisoning attack
                # dataset: Dp = {T1 ∩ tp}
                # model: the same type as the client’s model.
            self.server_dynamic_attack(self.clients)

            # Step4: Embedding Aggregation.
            self.server.poison_aggregation(self.clients, self.ent_freq_mat)

            logging.info('round: {} | loss: {:.4f}'.format(num_round, np.mean(round_loss)))
            self.write_training_loss(np.mean(round_loss), num_round)

            if num_round % self.args.check_per_round == 0 and num_round != 0:
                eval_res = self.evaluate()
                self.write_evaluation_result(eval_res, num_round)
                print('num_rououd:,',num_round)
                if eval_res['mrr'] > best_mrr:
                    best_mrr = eval_res['mrr']
                    best_epoch = num_round
                    logging.info('best model | mrr {:.4f}'.format(best_mrr))
                    self.save_checkpoint(num_round)
                    bad_count = 0
                else:
                    bad_count += 1
                    logging.info('best model is at round {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_epoch, best_mrr, bad_count))
            if bad_count >= self.args.early_stop_patience:
                logging.info('early stop at round {}'.format(num_round))
                break

        logging.info('finish training')
        logging.info('save best model')
        self.save_model(best_epoch)
        self.before_test_load()
        self.evaluate(istest=True)

    def before_test_load(self):
        state = torch.load(os.path.join(self.args.state_dir, self.args.name+ '.best'), map_location=self.args.gpu)
        self.server.ent_embed = state['ent_embed']
        self.server.rel_embed = state['server_rel_embed']
        self.server.victim_client=state['victim_client']
        for idx, client in enumerate(self.clients):
            client.rel_embed = state['rel_embed'][idx]

    def evaluate(self, istest=False,ispoisoned=False):
        self.send_emb()
        result = ddict(int)
        if istest:
            weights = self.test_eval_weights
        else:
            weights = self.valid_eval_weights

        if ispoisoned:
            with open(self.args.poisoned_triples_path+self.args.dataset_name + '_' + self.args.client_model + '_' + str(
            self.args.attack_entity_ratio)+ '_'+str(self.args.num_client) + '_poisoned_triples_dynamic_poisoned.json',
                      'r') as file1:
                poisoned_triples = json.load(file1)

            victim_client = list(poisoned_triples.keys())[0]
            common_difference = 256
            start_index = 0
            poisoned_tri = [poisoned_triples[victim_client][i] for i in
                            range(start_index, len(poisoned_triples[victim_client]), common_difference)]

            logging.info(
                '************ the test about poisoned triples in victim client **********' + str(victim_client))
            victim_client_res = self.clients[int(victim_client)].client_eval(poisoned_tri=poisoned_tri)
            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                victim_client_res['mrr'], victim_client_res['hits@1'],
                victim_client_res['hits@5'], victim_client_res['hits@10']))
            return victim_client_res

        logging.info('************ the test about poisoned datasets in all clients **********')
        for idx, client in enumerate(self.clients):
            client_res = client.client_eval(istest)

            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                client_res['mrr'], client_res['hits@1'],
                client_res['hits@5'], client_res['hits@10']))

            for k, v in client_res.items():
                result[k] += v * weights[idx]

        logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            result['mrr'], result['hits@1'],
            result['hits@5'], result['hits@10']))

        return result


