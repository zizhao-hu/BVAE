import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from model.ConvVAE import ConvVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HBVAE(ConvVAE):

    def __init__(self, name = 'VATE', norm = False, r=0, beta=1, C=0):
        super().__init__(name = name,r=r, beta=beta,C=C )
        self.norm = norm
        self.curlogvar = (torch.ones(16)*6).detach().to(device) 
        self.est_prior = 0.7*0.3   
        
    def le_score(self):
        weight = self.fc_mu.weight.data
        vec = nn.functional.normalize(weight, p=2, dim=1)
        vec = torch.pow(torch.mm(vec,vec.T),2)
        le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
        return le.item()
    
    def reparameterize(self, mu):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
       
        eps = torch.bernoulli(mu) # `randn_like` as we need the same size
        return eps
    def loss(self, x, reconstruction, mu, prior_mu=0):
        print(prior_mu)
        print(var_loss)
        recon_bce = nn.BCELoss(reduction='sum')(reconstruction, x)
        var_loss = nn.BCELoss(reduction='sum')(mu, prior_mu)
        print(mu)
        
       
        return recon_bce, var_loss
    def forward(self, x):
        # encoding
        est_mu, _ = self.encode(x)
        
        est_mu_01 = nn.functional.sigmoid(est_mu)
        agg_mu = torch.mean(est_mu_01, dim = 0)
     
        pri_mu = (agg_mu>0.5).float()*0.6+0.2
        print(pri_mu)
        print(agg_mu.cpu())
        # agg_logvar =  torch.log(est_mu_01*(1-est_mu_01))
        # est_logvar = torch.zeros_like(agg_logvar).fill_(self.est_logvar)
        z = self.reparameterize(pri_mu)
        reconstruction = self.decode(z)
        return reconstruction, pri_mu, agg_mu 

        
