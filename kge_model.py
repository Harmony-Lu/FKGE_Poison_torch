import torch
import torch.nn as nn


class KGEModel(nn.Module):
    def __init__(self, args, model_name):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )

    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        if not neg: # positive_samples
            head_idx = sample[:, 0]
            relation_idx = sample[:, 1]
            tail_idx = sample[:, 2]
            for idx in head_idx:
                entity_embedding[idx.item()]
            head = torch.index_select( # shape: 512x1x128
                entity_embedding, #14505x128
                dim=0,
                index=head_idx
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=relation_idx
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_idx
            ).unsqueeze(1)
        else: # negative_samples
            head_part, tail_part = sample
            batch_size = head_part.shape[0]


            head = torch.index_select( # shape: 512x1x128
                entity_embedding,
                dim=0,
                index=head_part[:, 0].long()
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1].long()
            ).unsqueeze(1)


            if tail_part == None:
                tail = entity_embedding.unsqueeze(0)
            else:
                negative_sample_size = tail_part.size(1)
                tail = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
            
        self.model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
        }
        # negative samples: head 512x1x128    relation 512x1x128      tail 512x256x128
        # positive_samples: head 512x1x128    relation 512x1x128      tail 512x1x128
        score = self.model_func[self.model_name](head, relation, tail)
        
        return score# 512x256
    
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
