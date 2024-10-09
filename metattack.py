import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from kge_model import KGEModel
from tqdm import tqdm
from process_data.generate_client_data_2 import TrainDataset, TensorTrainDataset
from torch.utils.data import TensorDataset

class BaseMeta(Module):
    def __init__(self, args, poisoned_triples, rel_embed, poisoned_ent_embed, victim_client_ent_embed):
        super(BaseMeta, self).__init__()
        self.args = args
        self.poisoned_triples = poisoned_triples
        # self.poisoned_triples = torch.tensor(poisoned_triples, dtype=torch.float32) # type: list
        # self.poisoned_triples = torch.tensor(poisoned_triples, dtype=torch.long) # Int型Tensor不能有梯度，有梯度必须float型tensor

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
        # self.kge_attack_model = KGEAttackModel(args, args.server_model)
        self.device = args.gpu
        
        # datasize = len(self.poisoned_triples) # 训练集中三元组数量
        # self.poison = Parameter(torch.Tensor(1,3))
        # self.poison.data.fill_(0) 


class Metattack(BaseMeta):
    def __init__(self, args, poisoned_triples, rel_embed, poisoned_ent_embed, victim_client_ent_embed):
        super(Metattack, self).__init__(args, poisoned_triples, rel_embed, poisoned_ent_embed, victim_client_ent_embed)
        # 初始化参数 rel_embedding和ent_embedding   已经初始化好了
    
    def forward(self):
        perturbations = 10
        # for j in tqdm(range(perturbations), desc="Perturbing graph"):
        # 每次注射一个有毒三元组到训练集中
        # self.poisoned_triples = torch.cat((self.poisoned_triples, self.poison), dim=0)
        train_dataset = TrainDataset(self.poisoned_triples, self.args.nentity, self.args.num_neg)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn,
        )

        sum_grad = 0
        for i in range(int(self.args.local_epoch)):
            for positive_sample, negative_sample, sample_idx in self.train_dataloader:

                # positive_sample = positive_sample.to(self.args.gpu)
                # negative_sample = negative_sample.to(self.args.gpu)

                # positive_sample = positive_sample.long()
                # negative_sample = negative_sample.long()
                negative_score = self.kge_model((positive_sample, negative_sample),
                                                self.rel_embed, self.poisoned_ent_embed)
                negative_score = (F.softmax(negative_score * float(self.args.adversarial_temperature), dim=1).detach()
                                * F.logsigmoid(-negative_score)).sum(dim=1) # shape: 512

                positive_score = self.kge_model(positive_sample,
                                                self.rel_embed, self.poisoned_ent_embed, neg=False)
                positive_score = F.logsigmoid(positive_score).squeeze(dim=1) # shape: 512

                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()
                loss = (positive_sample_loss + negative_sample_loss) / 2

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                # triple_grad = torch.autograd.grad(loss, self.poison)
                # print("triple_grad=", triple_grad.shape, triple_grad)
                if self.args.local_epoch==i+1:
                    # batch_grad = torch.autograd.grad(loss, positive_sample)[0]
                    # print('batch_grad=', batch_grad.shape, batch_grad)
                    if isinstance(sum_grad, int):
                        sum_grad = torch.autograd.grad(loss, self.poisoned_ent_embed)[0]
                    else:
                        sum_grad = sum_grad + torch.autograd.grad(loss, self.poisoned_ent_embed)[0]
            
        # 根据sum_grad对poisoned_triples进行投毒
        # 按行找到梯度绝对值之和最大的一行 作为argmax of meta gradients
        k = int(self.args.attack_entity_ratio) # 要攻击的entity数目：k
        row_grad_sum = torch.sum(torch.abs(sum_grad), dim=1) # shape: 14505
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
                attacked_ent_real_relation_list = [ attacked_ent_real_relation_list ]

            # (2) The server find the true tail entity of the attacked entity
            attacked_ent_real_tail_list = tail_list[ent_index]
            if type(attacked_ent_real_tail_list) is int:
                attacked_ent_real_tail_list = [ attacked_ent_real_tail_list ]

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