import torch
import torch.nn as nn

def le_score(weight):
    vec = nn.functional.normalize(weight, p=2, dim=1)
    vec = torch.pow(torch.mm(vec,vec.T),2)
    le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
    return le.item()

def kl(prior_mu, prior_logvar, mu, logvar):
    return -0.5 * torch.sum(1 + logvar-prior_logvar - (torch.abs(mu-prior_mu)).pow(2) - (logvar-prior_logvar).exp())