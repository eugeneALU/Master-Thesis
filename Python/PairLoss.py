import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, vector1, vector2, label):
        euclidean_distance = F.pairwise_distance(vector1, vector2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class CosineLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(CosineLoss, self).__init__()
        self.margin = margin

    def forward(self, vector1, vector2, label):
        cosine_distance = F.cosine_similarity(vector1, vector2)
        loss_contrastive = torch.mean((label) * torch.pow(cosine_distance, 2) +
                                (1-label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))


        return loss_contrastive