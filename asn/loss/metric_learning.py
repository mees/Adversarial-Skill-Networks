import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from asn.utils.comm import distance
from asn.utils.log import log

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    from:
        https://github.com/adambielski/siamese-triplet/blob/master/losses.py
    """

    def __init__(self, margin=.2):
        log.warning("use: torch.nn.TripletMarginLoss(margin=.2)!")
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss
        from: https://github.com/ChaofWang/Npair_loss_pytorch/blob/master/Npair_loss.py
    """

    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        '''
            target are class labels
        '''
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss


def _pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()




class LiftedStruct(nn.Module):
    ''' Lifted Structured Feature Embedding

    https://arxiv.org/abs/1511.06452
    form: https://github.com/vadimkantorov/metriclearningbench
    see also: https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4
    '''

    def forward(self, embeddings, labels, margin=1.0, eps=1e-4):
        loss = torch.zeros(1)
        if torch.cuda.is_available():
            loss = loss.cuda()
        loss = torch.autograd.Variable(loss)
        # L_{ij} = \log (\sum_{i, k} exp\{m - D_{ik}\} + \sum_{j, l} exp\{m - D_{jl}\}) + D_{ij}
        # L = \frac{1}{2|P|}\sum_{(i,j)\in P} \max(0, J_{i,j})^2
        d = _pdist(embeddings, squared=False, eps=eps)
        # pos mat  1 wehre labes are same for distane mat
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d)
                         for dim in [0, 1]]).type_as(d)

        neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
        loss += torch.sum(F.relu(pos.triu(1) * ((neg_i +
                                                 neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d))
        return loss


class LiftedStructPosMin(nn.Module):
    '''
        min pos distance untill in margin
    '''

    def forward(self, embeddings, labels, margin=1.0, eps=1e-4):
        loss = torch.zeros(1)
        if torch.cuda.is_available():
            loss = loss.cuda()
        loss = torch.autograd.Variable(loss)
        # L_{ij} = \log (\sum_{i, k} exp\{m - D_{ik}\} + \sum_{j, l} exp\{m - D_{jl}\}) + D_{ij}
        # L = \frac{1}{2|P|}\sum_{(i,j)\in P} \max(0, J_{i,j})^2
        d = _pdist(embeddings, squared=False, eps=eps)
        # pos mat  1 wehre labes are same for distane mat
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d)
                         for dim in [0, 1]]).type_as(d)

        neg = (pos<1).type_as(d)
        # pos_i = torch.mul((d-margin).exp(), 1 - neg).sum(1).expand_as(d)
        d=torch.mul(d, (d>margin).type_as(d))
        pos_i = torch.mul(d, 1 - neg).sum(1).expand_as(d)
        # loss += torch.sum(F.relu(neg.triu(1) * ((pos_i +
                                                 # pos_i.t()).log() + d)).pow(2)) / (neg.sum() - len(d))
        pos_dist_mean=torch.sum(pos_i)/ (neg.sum() - len(d))
        loss+=pos_dist_mean
        # print('pos_dist_mean: {}'.format(pos_dist_mean))
        return loss

class LiftedCombined(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=LiftedStructPosMin()
        self.b=LiftedStruct()

    def forward(self, embeddings, labels, margin=1.0, eps=1e-4):
        return self.b(embeddings, labels) + self.a(embeddings, labels,margin=1.2)

