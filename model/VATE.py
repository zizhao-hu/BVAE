import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from model.ConvVAE import ConvVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VATE(ConvVAE):

    def __init__(self, name = 'VATE', norm = False, r=0, beta=1, C=0):
        super().__init__(name = name,r=r, beta=beta,C=C )
        self.norm = norm
        self.curlogvar = (torch.ones(16)*6).detach().to(device)
    
    def loss(self, x, reconstruction, mu, logvar, prior_mu, prior_logvar):
        recon_bce = nn.BCELoss(reduction='sum')(reconstruction, x)
        var_loss =  -0.5 * torch.sum(1 + logvar-prior_logvar - (torch.abs(mu-prior_mu)-self.r).pow(2) - (logvar-prior_logvar).exp())
        if self.norm:
            norm_loss = self.le_score()
            return recon_bce, var_loss, norm_loss
        else:
            return 10*recon_bce, var_loss
        
    def le_score(self):
        weight = self.fc_mu.weight.data
        vec = nn.functional.normalize(weight, p=2, dim=1)
        vec = torch.pow(torch.mm(vec,vec.T),2)
        le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
        return le.item()
