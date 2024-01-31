import torch
from torch.nn import functional as F


# MAKE FUNCTION OF IT
def calculate_adjacency_matrix(features, temperature):
    """
    calculates the adjacency matrix of the given features.
    1. Calculate the pairwise Euclidean distances of the features
    2. Apply the temperature scaling
    """
    dist_matrix = torch.cdist(features, features, p=2)
    adj_matrix = F.softmax(-dist_matrix / temperature, dim=1)
    return adj_matrix


def manifold_matching_loss(image_features, text_features, temperature):
    """
    calculate the mm loss of the lico model
    """
    A_F = calculate_adjacency_matrix(image_features, temperature)
    A_G = calculate_adjacency_matrix(text_features, temperature)

    # Calculate the KL divergence loss for manifold matching
    loss = F.kl_div(A_G.log(), A_F, reduction='batchmean')
    return loss