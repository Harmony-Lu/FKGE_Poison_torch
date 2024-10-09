import torch
import torch.nn as nn
import numpy as np


class KGEAttackModel(nn.Module):
    def __init__(self, args, model_name):
        super(KGEAttackModel, self).__init__()
        self.model_name = model_name
        self.embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )

    # def forward(self, modified_adj, relation_embedding, entity_embedding, neg=True):
    def forward(self, triplet, relation_embedding, entity_embedding, neg=True):
        # nentity = modified_adj.shape[0]
        if not neg:
            # ht_idx = torch.nonzero(modified_adj)
            # dim0, dim1 = ht_idx.shape
            # triplet = []
            # for i in range(dim0):
            #     head_idx, tail_idx = ht_idx[i]
            #     relation_idx = int(modified_adj[head_idx][tail_idx]-1) if int(modified_adj[head_idx][tail_idx])>=1 else 0
            #     # head = entity_embedding[head_idx]
            #     # tail = entity_embedding[tail_idx]
            #     # relation = relation_embedding[relation_idx]
            #     triplet.append([head_idx, relation_idx, tail_idx])
            #     # triplet.append([head, relation, tail])
            # triplet = torch.tensor(triplet, dtype=torch.int32)

            # head = torch.index_select(
            #     entity_embedding,
            #     dim=0,
            #     index=triplet[:, 0]
            # ).unsqueeze(1)

            # relation = torch.index_select(
            #     relation_embedding,
            #     dim=0,
            #     index=triplet[:, 1]
            # ).unsqueeze(1)

            # tail = torch.index_select(
            #     entity_embedding,
            #     dim=0,
            #     index=triplet[:, 2]
            # ).unsqueeze(1)

            # head_idx = triplet[:,0]
            # relation_idx = triplet[:,1]
            # tail_idx = triplet[:,2]
            head = entity_embedding[triplet[:,0].long()].unsqueeze(1)       # 87035x1x128
            relation = relation_embedding[triplet[:,1].long()].unsqueeze(1) # 87035x1x128
            tail = entity_embedding[triplet[:,2].long()].unsqueeze(1)       # 87035x1x128

        self.model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }

        score = self.model_func[self.model_name](head, relation, tail)
        
        return score # 87035x1
    
    def TransE(self, head, relation, tail):

        score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score


    def DistMult(self, head, relation, tail):
        score = (head * relation) * tail
        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score
