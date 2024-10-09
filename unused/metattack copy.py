import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
# from gcn import GCN
from kge_model import KGEModel
from kge_attack_model import KGEAttackModel
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import utils
import math
import scipy.sparse as sp

import random
from process_data.generate_client_data_2 import TrainDataset, generate_new_id

class BaseMeta(Module):
    def __init__(self, args, poisoned_triples, rel_embed, poisoned_ent_embed, victim_client_ent_embed):
        super(BaseMeta, self).__init__()
        self.args = args
        # self.poisoned_triples = torch.tensor(poisoned_triples, dtype=torch.float32) # type: list
        self.poisoned_triples = poisoned_triples # type: list

        self.poisoned_ent_embed = poisoned_ent_embed
        self.rel_embed = rel_embed
        self.victim_client_ent_embed = victim_client_ent_embed

        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.train_iter = int(args.local_epoch)
        # self.surrogate_optimizer = optim.Adam([{'params': self.rel_embed}, # 刚初始化的rel_embed
        #                         {'params': self.victim_client_ent_embed}, # victim client ent_embed
        #                         {'params': self.poisoned_ent_embed}], lr=float(args.lr)) # 刚初始化的ent_embed
        
        self.optimizer = optim.Adam([{'params': self.rel_embed}, # 刚初始化的rel_embed
                                {'params': self.victim_client_ent_embed}, # victim client ent_embed
                                {'params': self.poisoned_ent_embed}], lr=float(args.lr)) # 刚初始化的ent_embed
        
        self.kge_model = KGEModel(args, args.server_model)
        self.kge_attack_model = KGEAttackModel(args, args.server_model)
        self.device = args.gpu

    # def train_surrogate(self):
    #     # 基于self.poison_triples训练正常的self.kge_model，得到训练后的ent_mebedding和rel_embedding
    #     train_dataset = TrainDataset(self.poisoned_triples, self.args.nentity, self.args.num_neg)
    #     train_dataloader = DataLoader(
    #         train_dataset,
    #         batch_size=self.args.batch_size,
    #         shuffle=True,
    #         collate_fn=TrainDataset.collate_fn
    #     )
    #     for i in range(self.train_iter):
    #         print('************** surrogate model training loss ******************')
    #         for batch in train_dataloader:
    #             positive_sample, negative_sample, sample_idx = batch

    #             positive_sample = positive_sample.to(self.args.gpu)
    #             negative_sample = negative_sample.to(self.args.gpu)

    #             negative_score = self.kge_model((positive_sample, negative_sample),
    #                                             self.rel_embed, self.poisoned_ent_embed)

    #             negative_score = (F.softmax(negative_score * float(self.args.adversarial_temperature), dim=1).detach()
    #                                 * F.logsigmoid(-negative_score)).sum(dim=1)

    #             positive_score = self.kge_model(positive_sample,
    #                                             self.rel_embed, self.poisoned_ent_embed, neg=False)

    #             positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

    #             positive_sample_loss = - positive_score.mean()
    #             negative_sample_loss = - negative_score.mean()
    #             loss = (positive_sample_loss + negative_sample_loss) / 2
    #             self.surrogate_optimizer.zero_grad()
    #             loss.backward()
    #             self.surrogate_optimizer.step()
    #     return self.rel_embed, self.poisoned_ent_embed # 返回正常kge_model基于未修改adj 的训练结果

class Metattack(BaseMeta):
    def __init__(self, args, poisoned_triples, rel_embed, poisoned_ent_embed, victim_client_ent_embed):
        super(Metattack, self).__init__(args, poisoned_triples, rel_embed, poisoned_ent_embed, victim_client_ent_embed)
        # 初始化参数 rel_embedding和ent_embedding   已经初始化好了

    def inner_train(self): # modified_poison_triples
        # 1 对于待训练参数设置：requires_grad=True      rel_embed&ent_embed已经设置了
        # 2 基于modified_adjacency与modified_poison_triples 训练
        # for i in range(self.train_iter):
        #     update model parameters and get updated rel_embed&ent_embed
        train_dataset = TrainDataset(self.poisoned_triples, self.args.nentity, self.args.num_neg)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )
        for i in range(self.train_iter):
            print('************** attack model training loss ******************')
            for batch in train_dataloader:
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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.rel_embed, self.poisoned_ent_embed
    
    # def get_meta_grad(self, modified_adj):
    def get_meta_grad(self):
        # # 一次training得到的loss作为attack_loss，output ent_embed&rel_embed
        # # adj_grad = autograd(attack_loss, self.adjacency_changes)
        
        # # # 根据inner_train训练得到的embedding计算loss
        # score = self.kge_attack_model( self.poisoned_triples, self.rel_embed, self.poisoned_ent_embed, neg=False)
        # batch_loss = -score.mean()
        # self.optimizer.zero_grad()
        # batch_loss.backward(retain_graph=True)

        # adj_grad = torch.autograd.grad(batch_loss, self.poisoned_ent_embed)[0]
        # self.optimizer.step()
        # # for group in self.optimizer.param_groups:
        # #     for p in group['params']:
        # #         print(p, p.grad)

        train_dataset = TrainDataset(self.poisoned_triples, self.args.nentity, self.args.num_neg)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )
        total_loss = 0.0
        batch_num = 0
        for batch in train_dataloader:
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
            batch_loss = (positive_sample_loss + negative_sample_loss) / 2
            
            self.optimizer.zero_grad()
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()
            total_loss += batch_loss
            batch_num += 1
            # losses.append(loss.item())
        
        # # attack_loss = torch.tensor(np.mean(losses))
        attack_loss = total_loss/batch_num
        print("attack_loss=", attack_loss)
        adj_grad = torch.autograd.grad(attack_loss, self.poisoned_ent_embed, retain_graph=True)[0]
        return adj_grad
    
    # def triplet2adjacency(self, triplet):
    #         # 初始化为0
    #         # triplet_adjacency_matrix = np.full((self.nrelation, self.nentity, self.nentity), 0)
    #         triplet_adjacency_matrix = np.zeros((self.nentity, self.nentity))
    #         for t in self.poisoned_triples:
    #             h = t[0].detach().numpy()
    #             r = t[1].detach().numpy()
    #             t = t[2].detach().numpy()
    #             triplet_adjacency_matrix[h][t] = r+1 # 矩阵的取值：1-237    0：两个entity之间没有任何关系 h-->t == r
    #             # triplet_adjacency_matrix[h][t] = r+1 # 矩阵的取值：1/0    0：两个entity之间没有任何关系     
    #             # triplet_adjacency_matrix[t][h] = r+1 # 矩阵的取值：1/0    0：两个entity之间没有任何关系 
    #         return torch.tensor(triplet_adjacency_matrix) # 二维Array
    
    # def forward(self, sample, relation_embedding, entity_embedding, neg=True):
    def forward(self, ori_poisoned_triples, perturbations):
        k = int(self.args.attack_entity_ratio) # 要攻击的entity数目：k
        self.inner_train()
        adj_meta_grad = self.get_meta_grad() # 相对于ent_embedding的梯度     
        # 按行找到梯度绝对值之和最大的一行 作为argmax of meta gradients
        row_grad_sum = torch.sum(torch.abs(adj_meta_grad), dim=1) # shape: 14505
        sorted_row_grad_sum, row_grad_sum_idx = torch.sort(row_grad_sum, descending=True)# 按降序排列
        # attacked_entity_list = row_grad_sum_idx[:k] # 找到对训练影响最大的10个entity
        # 把这k个entity作为攻击对象，生成其负样本插入self.poisoned_triples进行训练与测试，观察指标与性能变化
        poisoned_triples = []
        head_list = (np.array(self.poisoned_triples)[:,[0]]).squeeze().tolist()
        relation_list = (np.array(self.poisoned_triples)[:,[1]]).squeeze().tolist()
        tail_list = (np.array(self.poisoned_triples)[:,[2]]).squeeze().tolist()
        attack_entity_count = 0
        for ent in row_grad_sum_idx:
            if attack_entity_count>=k: # 寻找前k个可攻击的entity
                break
            if ent in head_list:
                ent_index = head_list.index(ent)
                attack_entity_count += 1
            else:
                print("not found this entity as head entity!")
                continue
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
            fake_r = np.random.choice(attacked_ent_fake_relation)
            # (6) The server randomly select the tail entity of the attacked entity
            fake_tail = np.random.choice(attacked_ent_fake_tail)

            # (7) The server generate the poisoned triplets for the attacked entity
            for i in range(256):
                poisoned_triples.append([int(ent),int(fake_r),int(fake_tail)])

        # 3.The poisoned triplets are added to dataset
        self.poisoned_triples = self.poisoned_triples + poisoned_triples  # t_p
        return self.poisoned_triples