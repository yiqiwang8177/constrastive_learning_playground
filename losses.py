import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, distance):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = self.distance(anchor, positive)  # .pow(.5)
        distance_negative = self.distance(anchor, negative)  # .pow(.5)
#         print('Pos:', distance_positive, '\tNeg:', distance_negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class HardNegativesBatchLoss(nn.Module):
    """
    Constastive loss
    create 2 matrices, find the worst pair of positive, and the worst pair of pos-neg. Margin-based loss
    """

    def __init__(self, margin, distance):
        super(HardNegativesBatchLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, positive, negative, size_average=True):
        # make sure positive/negative shape is: batch x dimension
        
        sim_pos = torch.cdist(positive, positive, p = 2).flatten() # 
        non_zero = torch.nonzero( sim_pos )
        loss1 = torch.max( sim_pos[non_zero] )
        
        sim_pos_neg = torch.cdist(positive, negative, p = 2).flatten() # 
        non_zero = torch.nonzero( sim_pos_neg )
        loss2 = torch.max( sim_pos_neg[non_zero] )
        
        losses = F.relu( loss1 - loss2 + self.margin)
        return losses

# class OnlineTripletLoss(nn.Module): # does not work yet
#     """
#     Online Triplets loss
#     Takes a batch of embeddings and corresponding labels.
#     Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
#     triplets
#     """

#     def __init__(self, margin, triplet_selector):
#         super(OnlineTripletLoss, self).__init__()
#         self.margin = margin
#         self.triplet_selector = triplet_selector

#     def forward(self, embeddings, target):

#         triplets = self.triplet_selector.get_triplets(embeddings, target)

#         if embeddings.is_cuda:
#             triplets = triplets.cuda()

#         ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
#         an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
#         losses = F.relu(ap_distances - an_distances + self.margin)

#         return losses.mean(), len(triplets)