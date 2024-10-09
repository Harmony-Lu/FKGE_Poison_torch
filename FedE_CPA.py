#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 22:05
# @Author  : zhixiuma
# @File    : fede.py
# @Project : FedE_Poison
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
# 追加的包
from itertools import permutations

# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from process_data.generate_client_data_2 import TrainDataset, generate_new_id
from random import choice


class Server(object):
    def __init__(self, args, nentity, clients):
        self.args = args
        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / int(args.hidden_dim)])
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
        
        self.clients = clients
        self.victim_client = None
        self.malicious_client = None

        self.poison_attack_state = 0 
        # 0     not start
        # 0->1     test all poisoned_triples
        # 1->2     get valid poisoned_triples, filter invalid poisoned_triples
        #       # inject poisoned_triples to malicious client and continue training
        # 2->3  
        self.poison_attack_round = None
        self.poison_attack_interval = 5 # 间隔5个round，再计算poisoned_triples的score
        self.poisoned_triples = []
        self.poisoned__triples_score = []

    def send_emb(self):
        return copy.deepcopy(self.ent_embed)
    
    def begin_random_poisoning_attack(self, current_round, malicious_client=1, victim_client=0):
        if self.poison_attack_state==0:
            self.malicious_client = malicious_client
            self.victim_client = victim_client

            self.poison_attack_state = 1
            self.poison_attack_round = current_round
            # get overlap entity between malicious_client(m_c) and victim client(v_c)
            victim_head_list = (np.array(self.clients[victim_client].train_dataloader.dataset.triples)[:, 0]).squeeze().tolist()
            victim_tail_list = (np.array(self.clients[victim_client].train_dataloader.dataset.triples)[:, 2]).squeeze().tolist()
            malicious_head_list = (np.array(self.clients[malicious_client].train_dataloader.dataset.triples)[:, 0]).squeeze().tolist()
            malicious_tail_list = (np.array(self.clients[malicious_client].train_dataloader.dataset.triples)[:, 2]).squeeze().tolist()
            malicious_relation_list = (np.array(self.clients[malicious_client].train_dataloader.dataset.triples)[:, 1]).squeeze().tolist()
            
            overlap_head_list = list(set(victim_head_list).intersection(set(malicious_head_list)))
            overlap_tail_list = list(set(victim_tail_list).intersection(set(malicious_tail_list)))
            print("overlap head", len(set(victim_head_list)), len(set(malicious_head_list)), len(overlap_head_list))
            print("overlap tail", len(set(victim_tail_list)), len(set(malicious_tail_list)), len(overlap_tail_list))

            # calculate m_c score of all possible poisoned_triples( h,t: v_c embedding  r: m_c embedding)
            # 1 the malicious client randomly select the index of the attacked entities (head) from the overlap.
            num_attacked_entities = int(self.args.attack_entity_ratio)
            attacked_entity_list = random.sample(overlap_head_list, k=num_attacked_entities)

            # 2 The poisoned triplets are generated based on the attacked entities
            for ent in attacked_entity_list:
                ent_index = malicious_head_list.index(ent) # 首先, 保证poisoned_triple不存在于malicious_client
                # (1) the malicious client find the false relation of the attacked entity
                attacked_ent_real_relation_list = malicious_relation_list[ent_index]
                if type(attacked_ent_real_relation_list) is int:
                    attacked_ent_real_relation_list = [attacked_ent_real_relation_list]
                attacked_ent_fake_relation = list(set(malicious_relation_list) - set(attacked_ent_real_relation_list))
                # (2) the malicious client find the false tail entity of the attacked entity
                attacked_ent_real_tail_list = malicious_tail_list[ent_index]
                if type(attacked_ent_real_tail_list) is int:
                    attacked_ent_real_tail_list = [attacked_ent_real_tail_list]
                attacked_ent_fake_tail = list(set(malicious_tail_list) - set(attacked_ent_real_tail_list))
                # ensure 'the fake tail ∈ the overlap'
                attacked_ent_maliciou_fake_tail = list(set(attacked_ent_fake_tail).intersection(overlap_tail_list))

                # (5) The server randomly select the false relation of the attacked entity
                fake_r = choice(attacked_ent_fake_relation)
                # (6) The server randomly select the tail entity of the attacked entity
                fake_tail = choice(attacked_ent_maliciou_fake_tail)
                # (7) The server generate the poisoned triplets for the attacked entity
                triple = [int(ent),int(fake_r),int(fake_tail)]
                score = self.clients[malicious_client].kge_model(torch.IntTensor([triple]),
                                        self.clients[malicious_client].rel_embed, 
                                        self.clients[victim_client].ent_embed, neg=False)
                self.poisoned_triples.append(triple)
                self.poisoned__triples_score.append(score.item())

    def random_poisoning_attack(self, current_round):
        if self.poison_attack_state==0:
            return
        if self.poison_attack_state==1 and self.poison_attack_interval+self.poison_attack_round==current_round:
            # 对比方法 directly inject poisoned_triples to malicious client train dataset
            poisoned_triples = []
            for triple in self.poisoned_triples:
                for i in range(256):
                    poisoned_triples.append(triple)
            malicious_client_train_dataset = self.clients[self.malicious_client].train_dataloader.dataset.triples
            poisoned_dataset =  malicious_client_train_dataset + poisoned_triples
            print("making poisoned dataset...", len(malicious_client_train_dataset), len(poisoned_triples), len(poisoned_dataset))
            train_dataset = TrainDataset(poisoned_dataset, self.args.nentity, self.args.num_neg)
            self.clients[self.victim_client].train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            )
            self.poison_attack_state = 2

    def begin_inference_based_poisoning_attack(self, current_round, malicious_client=1, victim_client=0):
        if self.poison_attack_state==0:
            self.malicious_client = malicious_client
            self.victim_client = victim_client

            self.poison_attack_state = 1
            self.poison_attack_round = current_round
            # get overlap entity between malicious_client(m_c) and victim client(v_c)
            victim_head_list = (np.array(self.clients[victim_client].train_dataloader.dataset.triples)[:, 0]).squeeze().tolist()
            victim_tail_list = (np.array(self.clients[victim_client].train_dataloader.dataset.triples)[:, 2]).squeeze().tolist()
            malicious_head_list = (np.array(self.clients[malicious_client].train_dataloader.dataset.triples)[:, 0]).squeeze().tolist()
            malicious_tail_list = (np.array(self.clients[malicious_client].train_dataloader.dataset.triples)[:, 2]).squeeze().tolist()
            malicious_relation_list = (np.array(self.clients[malicious_client].train_dataloader.dataset.triples)[:, 1]).squeeze().tolist()
            
            overlap_head_list = list(set(victim_head_list).intersection(set(malicious_head_list)))
            overlap_tail_list = list(set(victim_tail_list).intersection(set(malicious_tail_list)))
            print("overlap head", len(set(victim_head_list)), len(set(malicious_head_list)), len(overlap_head_list))
            print("overlap tail", len(set(victim_tail_list)), len(set(malicious_tail_list)), len(overlap_tail_list))

            # calculate m_c score of all possible poisoned_triples( h,t: v_c embedding  r: m_c embedding)
            # 1 the malicious client randomly select the index of the attacked entities (head) from the overlap.
            num_attacked_entities = int(self.args.attack_entity_ratio)
            attacked_entity_list = random.sample(overlap_head_list, k=num_attacked_entities)

            # 2 The poisoned triplets are generated based on the attacked entities
            for ent in attacked_entity_list:
                ent_index = malicious_head_list.index(ent) # 首先, 保证poisoned_triple不存在于malicious_client
                # (1) the malicious client find the false relation of the attacked entity
                attacked_ent_real_relation_list = malicious_relation_list[ent_index]
                if type(attacked_ent_real_relation_list) is int:
                    attacked_ent_real_relation_list = [attacked_ent_real_relation_list]
                attacked_ent_fake_relation = list(set(malicious_relation_list) - set(attacked_ent_real_relation_list))
                # (2) the malicious client find the false tail entity of the attacked entity
                attacked_ent_real_tail_list = malicious_tail_list[ent_index]
                if type(attacked_ent_real_tail_list) is int:
                    attacked_ent_real_tail_list = [attacked_ent_real_tail_list]
                attacked_ent_fake_tail = list(set(malicious_tail_list) - set(attacked_ent_real_tail_list))
                # ensure 'the fake tail ∈ the overlap'
                attacked_ent_maliciou_fake_tail = list(set(attacked_ent_fake_tail).intersection(overlap_tail_list))

                # (5) The server randomly select the false relation of the attacked entity
                fake_r = choice(attacked_ent_fake_relation)
                # (6) The server randomly select the tail entity of the attacked entity
                fake_tail = choice(attacked_ent_maliciou_fake_tail)
                # (7) The server generate the poisoned triplets for the attacked entity
                triple = [int(ent),int(fake_r),int(fake_tail)]
                score = self.clients[malicious_client].kge_model(torch.IntTensor([triple]),
                                        self.clients[malicious_client].rel_embed, 
                                        self.clients[victim_client].ent_embed, neg=False)
                self.poisoned_triples.append(triple)
                self.poisoned__triples_score.append(score.item())

    def inference_based_poisoning_attack(self, current_round):
        if self.poison_attack_state==0:
            return
        if self.poison_attack_state==1 and self.poison_attack_interval+self.poison_attack_round==current_round:
            # our method
            # calculate m_c score of poisoned_triples ( h,t: v_c embedding  r: m_c embedding)
            # filter valid poisoned_triples
                # if score>score_before 表明该poisoned_triple在victim client中很可能存在 不合格
                # else 该poisoned_triple 合格
            # inject valid poisoned_triples to client1 train_dataset

            # 对比方法 directly inject poisoned_triples to malicious client train dataset
            poisoned_triples = []
            for triple in self.poisoned_triples:
                for i in range(256):
                    poisoned_triples.append(triple)
            malicious_client_train_dataset = self.clients[self.malicious_client].train_dataloader.dataset.triples
            poisoned_dataset =  malicious_client_train_dataset + poisoned_triples
            print("making poisoned dataset...", len(malicious_client_train_dataset), len(poisoned_triples), len(poisoned_dataset))
            train_dataset = TrainDataset(poisoned_dataset, self.args.nentity, self.args.num_neg)
            self.clients[self.victim_client].train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            )

            self.poison_attack_state = 2
    

    # 原始聚合
    def aggregation(self, clients, ent_update_weights):
        agg_ent_mask = ent_update_weights
        agg_ent_mask[ent_update_weights != 0] = 1

        ent_w_sum = torch.sum(agg_ent_mask, dim=0)  # ent_w_sum: tensor([3., 1., 3.,  ..., 1., 1., 1.], device='cuda:0')
        ent_w = agg_ent_mask / ent_w_sum
        ent_w[torch.isnan(ent_w)] = 0
        if self.args.client_model in ['RotatE', 'ComplEx']:
            update_ent_embed = torch.zeros(self.nentity, int(self.args.hidden_dim) * 2).to(self.args.gpu)
        else:
            update_ent_embed = torch.zeros(self.nentity, int(self.args.hidden_dim)).to(self.args.gpu)

        for i, client in enumerate(clients):
            local_ent_embed = client.ent_embed.clone().detach()
            update_ent_embed += local_ent_embed * ent_w[i].reshape(-1, 1)
        self.ent_embed = update_ent_embed.requires_grad_()

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
            # print(poisoned_tri)
            poisoned_tri = np.array(poisoned_tri).astype(int)
            head_idx, rel_idx, tail_idx = poisoned_tri[:, 0],poisoned_tri[:, 1], poisoned_tri[:, 2]
            labels = self.poisoned_labels(poisoned_tri)

            pred = self.kge_model((torch.IntTensor(poisoned_tri.astype(int)).to(self.args.gpu), None),
                                  self.rel_embed, self.ent_embed)
            b_range = torch.arange(pred.size()[0], device=self.args.gpu)
            target_pred = pred[b_range, tail_idx]
            pred = torch.where(torch.FloatTensor(labels).byte().to(self.args.gpu), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred
            # 按行排列，返回排序索引
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]
            ranks = ranks.float()
            count = torch.numel(ranks)
            print(ranks)
            results['count'] += count
            results['mr'] += torch.sum(ranks).item()/len(poisoned_tri)
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
class FedE_CPA(object):
    def __init__(self, args, all_data):
        self.args = args

        self.malicious_client = 1
        self.victim_client = 0
        # self.victim_client = None # Server as victim

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

        self.server = Server(args, nentity, self.clients)

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
                 'rel_embed': [client.rel_embed for client in self.clients]}

        for filename in os.listdir(self.args.state_dir):
            if self.args.name in filename.split('.') and os.path.isfile(os.path.join(self.args.state_dir, filename)):
                os.remove(os.path.join(self.args.state_dir, filename))
        # save current checkpoint
        torch.save(state, os.path.join(self.args.state_dir,
                                       self.args.name + '.' + str(e) + '.ckpt'))

    def save_model(self, best_epoch):
        os.rename(os.path.join(self.args.state_dir,  self.args.name+ '.' + str(best_epoch) + '.ckpt'),
                  os.path.join(self.args.state_dir, self.args.name+ '.best'))

    def send_emb(self):
        for k, client in enumerate(self.clients):
            client.ent_embed = self.server.send_emb()

    def train(self):
        best_epoch = 0
        best_mrr = 0
        bad_count = 0
        for num_round in range(self.args.max_round):
            n_sample = max(round(self.args.fraction * self.num_clients), 1)
            sample_set = np.random.choice(self.num_clients, n_sample, replace=False)

            self.send_emb()
            if num_round==0: # 从第2轮训练开始，进行attack
                if self.args.poisoning_attack_type=='random':
                    self.server.begin_random_poisoning_attack(num_round, self.malicious_client, self.victim_client)
                elif self.args.poisoning_attack_type=='inference_based':
                    self.server.begin_inference_based_poisoning_attack(num_round, self.malicious_client, self.victim_client)

            if self.args.poisoning_attack_type=='random':
                self.server.random_poisoning_attack(num_round)
            elif self.args.poisoning_attack_type=='inference_based':
                self.server.inference_based_poisoning_attack(num_round)

            round_loss = 0
            for k in iter(sample_set):
                # client_loss = self.clients[k].client_update()
                client_loss = 1.0
                round_loss += client_loss
            round_loss /= n_sample

            self.server.aggregation(self.clients, self.ent_freq_mat)

            # 评估 victim-client的模型性能所受影响
            # 评估 所有clients的模型性能所受影响

            logging.info('round: {} | loss: {:.4f}'.format(num_round, np.mean(round_loss)))
            self.write_training_loss(np.mean(round_loss), num_round)

            if num_round % self.args.check_per_round == 0 and num_round != 0:
                eval_res = self.evaluate()
                self.write_evaluation_result(eval_res, num_round)

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
            with open(self.args.poisoned_triples_path+self.args.dataset_name + '_' + self.args.client_model +'_' + str(
            self.args.attack_entity_ratio) + '_'+str(self.args.num_client) + '_poisoned_triples_static_poisoned.json',
                      'r') as file1:
                poisoned_triples = json.load(file1)
            
            victim_client = list(poisoned_triples.keys())[0]
            # print(poisoned_triples[victim_client])
            common_difference = 256
            start_index = 0
            poisoned_tri = [poisoned_triples[victim_client][i] for i in
                            range(start_index, len(poisoned_triples[victim_client]), common_difference)]
            logging.info('************ the test about poisoned triples in victim client **********' + str(victim_client))
            victim_client_res = self.clients[int(victim_client)].client_eval(poisoned_tri=poisoned_tri)
            logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
                victim_client_res['mrr'], victim_client_res['hits@1'],
                victim_client_res['hits@5'], victim_client_res['hits@10']))

            return victim_client_res

        logging.info('************ the test about poisoned datasets in all clients CPA **********')
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


