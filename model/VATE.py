import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from ... import ConvVAE

class VATE(ConvVAE):

    def __init__(self, norm = False):
        super().__init__(norm)
        self.norm = norm
        self.curvar = torch.ones(16)
    
    def loss(self, x, reconstruction, mu, log_var):
        recon_bce = nn.BCELoss(reduction='sum')(reconstruction, x)
        var_loss =  -0.5 * torch.sum(1 + log_var - (torch.abs(mu)-self.r).pow(2) - log_var.exp())
        if self.norm:
            norm_loss = self.le_score()
            return recon_bce, var_loss, norm_loss
        else:
            return recon_bce, var_loss
        
    def le_score(self):
        weight = self.fc_mu.weight.data
        vec = nn.functional.normalize(weight, p=2, dim=1)
        vec = torch.pow(torch.mm(vec,vec.T),2)
        le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
        return le.item()


from torchsummary import summary
model = VATE(True)
print(summary(model, (3, 64,64)))