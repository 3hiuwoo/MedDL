import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations


def id_contrastive_loss(views, id):
    ''' Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        views (torch.Tensor): embeddings from model for different perturbations of same instance (NxBxH)
        id (list): ids of instances in batch
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    '''
    id = id.cpu().detach().numpy() 
    interest_matrix = np.equal.outer(id, id).astype(int) # B x B
    # only consider upper diagnoal where the queue is taken into account
    rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
    rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs

    nviews = set(range(views.shape[0]))
    view_combinations = combinations(nviews, 2)
    loss = 0
    ncombinations = 0
    temperature = 0.1
    eps = 1e-12
    for combination in view_combinations:
        view1 = views[combination[0], ...] # B x H
        view2 = views[combination[1], ...] # B x H
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        sim_matrix = torch.mm(view1, view2.transpose(0,1)) # B x B
        
        argument = sim_matrix / temperature
        sim_matrix_exp = torch.exp(argument)
        
        triu_elements = sim_matrix_exp[rows1,cols1]
        tril_elements = sim_matrix_exp[rows2,cols2]
        diag_elements = torch.diag(sim_matrix_exp)
        
        triu_sum = torch.sum(sim_matrix_exp,1)
        tril_sum = torch.sum(sim_matrix_exp,0)
        
        loss_diag1 = -torch.mean(torch.log((diag_elements+eps)/(triu_sum+eps)))
        loss_diag2 = -torch.mean(torch.log((diag_elements+eps)/(tril_sum+eps)))
        
        loss_triu = -torch.mean(torch.log((triu_elements+eps)/(triu_sum[rows1]+eps)))
        loss_tril = -torch.mean(torch.log((tril_elements+eps)/(tril_sum[cols2]+eps)))
        
        loss = loss_diag1 + loss_diag2
        loss_terms = 2

        if len(rows1) > 0:
            loss += loss_triu #technically need to add 1 more term for symmetry
            loss_terms += 1
        
        if len(rows2) > 0:
            loss += loss_tril #technically need to add 1 more term for symmetry
            loss_terms += 1
    
        ncombinations += 1
        
    loss = loss /(loss_terms * ncombinations)
    
    return loss


def id_momentum_loss(q, k, queue, id, id_queue):
    ''' Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        q (torch.Tensor): query embeddings from model for different perturbations of same instance (NxBxH)
        k (torch.Tensor): key embeddings from model for different perturbations of same instance (NxBxH)
        queue (torch.Tensor): queue embeddings from model for different perturbations of same instance (NxBxH)
        id (list): ids of instances in batch
        id_queue (torch.Tensor): queue ids
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    '''
    id = id.cpu().detach().numpy()
    id_queue = id_queue.cpu().detach().numpy()
    batch_interest_matrix = np.equal.outer(id, id).astype(int) # B x B
    queue_interest_matrix = np.equal.outer(id, id_queue).astype(int) # B x K
    interest_matrix = np.concatenate((batch_interest_matrix, queue_interest_matrix), axis=1) # B x (B+K)
    # only consider upper diagnoal where the queue is taken into account
    rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
    # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs

    loss = 0
    temperature = 0.1
    eps = 1e-12
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    batch_sim_matrix = torch.mm(q, k.t()) # B x B
    queue_sim_matrix = torch.mm(q, queue.t()) # B x K
    sim_matrix = torch.cat((batch_sim_matrix, queue_sim_matrix), dim=1) # B x (B+K)
    argument = sim_matrix / temperature
    sim_matrix_exp = torch.exp(argument)
    
    diag_elements = torch.diag(sim_matrix_exp)
    triu_elements = sim_matrix_exp[rows1,cols1]
    
    loss_diag = -torch.mean(torch.log((diag_elements+eps)/(torch.sum(sim_matrix_exp,1)+eps)))
    loss_triu = -torch.mean(torch.log((triu_elements+eps)/(torch.sum(sim_matrix_exp,1)[rows1]+eps)))
    
    loss = loss_diag + loss_triu
    
    return loss
    
    
        
