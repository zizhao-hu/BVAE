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
        
    def le_score(self):
        weight = self.fc_mu.weight.data
        vec = nn.functional.normalize(weight, p=2, dim=1)
        vec = torch.pow(torch.mm(vec,vec.T),2)
        le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
        return le.item()
    
    def forward(self, x):
        # encoding
        est_mu, est_logvar = self.encode(x)
        agg_mu = torch.mean(est_mu, dim = 0)
        agg_logvar =  torch.log(torch.var(est_mu, dim = 0))
        print(agg_logvar)
        print(est_logvar)
        # get the latent vector through reparameterization
        z = self.reparameterize(agg_mu, agg_logvar)
        reconstruction = self.decode(z)
        return reconstruction, est_mu, est_logvar, agg_mu, agg_logvar 

        
