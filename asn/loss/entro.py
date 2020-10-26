
import torch
import torch.nn as nn
from torch.nn import functional as F


class HLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b/x.size(0)

def marginalized_entropy(y):
    y=F.softmax(y, dim=1)
    y1 = y.mean(0)
    y2 = -torch.sum(y1*torch.log(y1+1e-6))
    return y2

def entropy(y):
    y1 = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
    y2 = 1./y.size(0)*y1.sum()
    return y2
