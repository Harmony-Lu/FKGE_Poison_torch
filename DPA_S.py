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


class Server(object):
    def __init__(self, args, nentity):
        self.args = args
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / int(args.hidden_dim)])

        # the server initializes a global entity embeddings matrix randomly
        if args.client_model in ['RotatE', 'ComplEx']:
            self.ent_embed = torch.zeros(nentity, int(args.hidden_dim) * 2).to(args.gpu).requires_grad_()
        else:
            self.ent_embed = torch.zeros(nentity, int(args.hidden_dim)).to(args.gpu).requires_grad_()

        nn.init.uniform_(
            tensor=self.ent_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        self.nentity = nentity

        self.train_dataloader = None
        self.victim_client = None
        self.kge_model = KGEModel(args, args.server_model)

    # the TransE model used to perform inference attacks,
    def TransE(self, head, tail):
        r = head - tail
        return r

    def send_emb(self):
        return copy.deepcopy(self.ent_embed)

    def poison_attack_random(self,  victim_client=0):
        self.victim_client=victim_client
        # Step1: Relation Inference
        # 0.Suppose the server knows the victim client's training triplet
        with open('process_data/client_data/' + self.args.dataset_name + '_' + str(self.args.num_client) + '_with_new_id.json', 'r') as file1:
            real_triples = json.load(file1)

        head_list = (np.array(real_triples[victim_client]['train'])[:,[0]]).squeeze().tolist()
        relation_list = (np.array(real_triples[victim_client]['train'])[:, [1]]).squeeze().tolist()
        tail_list =(np.array(real_triples[victim_client]['train'])[:, [2]]).squeeze().tolist()

        if int(self.args.attack_entity_ratio)==0:
            self.poisoned_triples = real_triples[victim_client]['train']
            print('len(self.poisoned_triples:',len(self.poisoned_triples))
            return

        # 1.The server randomly select the index of the attacked entities (head) from the victim client.
        num_attacked_entities = int(self.args.attack_entity_ratio)
        attacked_entity_mask = np.random.choice(len(head_list), num_attacked_entities, replace=False)
        attacked_entity_list = np.array(head_list)[attacked_entity_mask]

        #  2.The poisoned triplets are generated based on the attacked entities
        poisoned_triples = []
        for ent in attacked_entity_list:
            ent_index = head_list.index(ent)
            # (1) The server find the true relation of the attacked entity
            attacked_ent_real_relation_list = relation_list[ent_index]
            # print('attacked_ent_real_relation_list:',attacked_ent_real_relation_list)
            if type(attacked_ent_real_relation_list) is int:
                attacked_ent_real_relation_list = [attacked_ent_real_relation_list]

            # (2) The server find the true tail entity of the attacked entity
            attacked_ent_real_tail_list = tail_list[ent_index]
            if type(attacked_ent_real_tail_list) is int:
                attacked_ent_real_tail_list = [attacked_ent_real_tail_list]

            # (3) The server find the false relation of the attacked entity
            attacked_ent_fake_relation = list(set(relation_list) - set(attacked_ent_real_relation_list))
            # (4) The server find the false tail entity of the attacked entity
            attacked_ent_fake_tail = list(set(tail_list) - set(attacked_ent_real_tail_list))

            # Step2: Poison Data Generation.
            # (5) The server randomly select the false relation of the attacked entity
            fake_r = choice(attacked_ent_fake_relation)
            # (6) The server randomly select the tail entity of the attacked entity
            fake_tail = choice(attacked_ent_fake_tail)

            # (7) The server generate the poisoned triplets for the attacked entity
            for i in range(256):
                poisoned_triples.append([int(ent),int(fake_r),int(fake_tail)])

        # 3.The poisoned triplets are saved in file
        self.poisoned_tri = poisoned_triples  # t_p
        dic = {}
        dic[self.victim_client] = self.poisoned_tri
        if not os.path.exists(self.args.poisoned_triples_path):
            os.makedirs(self.args.poisoned_triples_path)
        with open(
                self.args.poisoned_triples_path + self.args.dataset_name + '_' + self.args.client_model + '_' + str(
                        self.args.attack_entity_ratio) + '_'+str(self.args.num_client) +'_poisoned_triples_dynamic_poisoned.json', 'w') as file1:
            json.dump(dic, file1)

        # 4、The server generate the training dataset D_p = {T_1 + t_p}
        # real_triples[victim_client]['train']: T_1
        # poisoned_triples: t_p
        self.poisoned_triples = poisoned_triples + real_triples[victim_client]['train']

        print(len(self.poisoned_triples))


    def create_poison_dataset(self):
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
        else:
            self.rel_embed = torch.zeros(self.args.nrelation, int(self.args.hidden_dim)).to(self.args.gpu).requires_grad_()

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
    def train_poison_model(self,clients):
        if self.victim_client == None:
            self.victim_client = 0
        else:
            self.victim_client = self.victim_client

        self.victim_client_ent_embed = clients[self.victim_client].ent_embed

        if self.train_dataloader == None: # 初始化self.poisoned_triples：生成poisoned_triples
            self.poison_attack_random(victim_client=self.victim_client) # 投毒过程

        self.create_poison_dataset() # 初始化self.rel_embed
        client_loss = self.server_poison_dynamic_update() # training of shadow model
        

    # The Dynamic Poisoning Attack
    def server_poison_dynamic_update(self):
        self.poisoned_ent_embed = self.send_emb()
        optimizer = optim.Adam([{'params': self.rel_embed}, # 刚初始化的rel_embed
                                {'params': self.victim_client_ent_embed}, # victim client ent_embed
                                {'params': self.poisoned_ent_embed}], lr=float(self.args.lr)) # 刚初始化的ent_embed
        losses = []
        head = np.array(self.poisoned_tri)[:, 0] # 制造的有毒三元组的头尾实体
        tail = np.array(self.poisoned_tri)[:, 2]
        for i in range(int(self.args.local_epoch)):
            # the training dataset(D_p = T_1 + t_p) + shadow model
            # print('************** server training loss ******************')
            for batch in self.train_dataloader:
                positive_sample, negative_sample, sample_idx = batch

                positive_sample = positive_sample.to(self.args.gpu)
                negative_sample = negative_sample.to(self.args.gpu)

                negative_score = self.kge_model((positive_sample, negative_sample),
                                                self.rel_embed, self.poisoned_ent_embed)

                negative_score = (F.softmax(negative_score * float(self.args.adversarial_temperature), dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

                positive_score = self.kge_model(positive_sample,
                                                self.rel_embed, self.poisoned_ent_embed, neg=False)

                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()
                loss = (positive_sample_loss + negative_sample_loss) / 2

                victim_client_head_emb = self.victim_client_ent_embed[head].unsqueeze(dim=1) # victim client embedding
                victim_client_tail_emb = self.victim_client_ent_embed[tail].unsqueeze(dim=1)
                server_head_emb = self.poisoned_ent_embed[head].unsqueeze(dim=1)# shadow model trained embedding
                server_tail_emb = self.poisoned_ent_embed[tail].unsqueeze(dim=1)

                # minimize the embedding inconsistencies between the server and the victim client
                loss_dis = ((victim_client_head_emb - server_head_emb) + (
                            victim_client_tail_emb - server_tail_emb)).mean()
                
                # loss = loss + loss_dis
                # loss = loss +0.000001*loss_dis
                loss = loss

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
        self.ent_freq_mat, rel_embed_list, nentity, nrelation,all_ent_list = read_triples(all_data, args)

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
                client_loss = self.clients[k].client_update()
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
            if num_round%15==0 and num_round != 0:
                self.evaluate(istest=True)


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

    def evaluate(self, istest=False, ispoisoned=False):
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

        logging.info('************ the test about poisoned datasets in all clients DPA_S ')
        if istest:
            logging.info('TEST MODE')
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


