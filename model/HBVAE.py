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
        self.est_logvar = 0.7*0.3   
        
    def le_score(self):
        weight = self.fc_mu.weight.data
        vec = nn.functional.normalize(weight, p=2, dim=1)
        vec = torch.pow(torch.mm(vec,vec.T),2)
        le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
        return le.item()
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
       
        eps = torch.bernoulli(mu) # `randn_like` as we need the same size
        return eps
    def loss(self, x, reconstruction, mu, logvar, prior_mu=0, prior_logvar=0):
        recon_bce = nn.BCELoss(reduction='sum')(reconstruction, x)
        var_loss = nn.BCELoss(reduction='sum')(mu, prior_mu)
        return recon_bce, var_loss
    def forward(self, x):
        # encoding
        est_mu, _ = self.encode(x)
        est_mu_01 = nn.functional.softmax(est_mu)
        agg_mu = torch.mean(est_mu_01, dim = 0)
        agg_logvar =  torch.log(est_mu_01*(1-est_mu_01))
        est_logvar = torch.zeros.fill(est_logvar)
        z = self.reparameterize(est_mu, est_logvar)
        reconstruction = self.decode(z)
        return reconstruction, est_mu_01, est_logvar, agg_mu, agg_logvar 

        
