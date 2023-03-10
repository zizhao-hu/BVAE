import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from model.ConvVAE import ConvVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HBVAE(ConvVAE):

    def __init__(self, name = 'HBVAE', norm = False, r=0, beta=1, C=0):
        super().__init__(name = name,r=r, beta=beta,C=C )
        self.norm = norm
        rand = torch.rand(32).reshape(1,32)
        sorted,_ = rand.sort()
        self.prior = sorted.expand(64,32).to(device).detach()

        
    def le_score(self):
        weight = self.fc_mu.weight.data
        vec = nn.functional.normalize(weight, p=2, dim=1)
        vec = torch.pow(torch.mm(vec,vec.T),2)
        le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
        return le.item()
    
    def reparameterize(self, mu, n_trials):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        eps = torch.zeros_like(mu)
        for i in range(n_trials):
            eps = (eps*i + torch.bernoulli(mu))/i+1 # `randn_like` as we need the same size
        return eps
    def loss(self, x, reconstruction, mu):
        recon_bce = nn.BCELoss(reduction='sum')(reconstruction, x)

        print(mu)
        print(self.prior)
        var_loss = nn.BCELoss(reduction='mean')(mu, self.prior)
        print(var_loss)
        
       
        return recon_bce, var_loss
    def forward(self, x):
        # encoding
        est_mu, _ = self.encode(x)
        
        est_mu = torch.sigmoid(est_mu) 
        rep_mu = self.reparameterize(est_mu, 20)
        z = torch.logit(rep_mu)
        reconstruction = self.decode(z)
        return reconstruction, est_mu

        
