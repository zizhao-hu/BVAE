import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F


### model configs
kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 3 
latent_dim = 32 # latent dimension for sampling
###

class ConvVAE(nn.Module):
    # beta-VAE, Disentanling beta-VAE, binarized-VAE
    def __init__(self, name="ConvVAE", beta = 1, C = 0, r = 0, norm=False):
        super(ConvVAE, self).__init__()
        self.name = name
        self.norm = False
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3d5 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*8, out_channels=128, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3d5 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # self.dec4 = nn.ConvTranspose2d(
        #     in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
        #     stride=2, padding=1
        # )
        self.beta = beta
        self.C = C
        self.r = r

    def encode(self, x):
         # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc3d5(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def decode(self, z):
        z = self.fc2(z)
        z = z.view(-1, 128, 1, 1)
 
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec3d5(x))
        reconstruction = torch.sigmoid(self.dec4(x))

        return reconstruction
    
    def loss(self, x, reconstruction, mu, logvar, prior_mu=0, prior_logvar=0):
        recon_bce = nn.BCELoss(reduction='sum')(reconstruction, x)
        var_loss =  -0.5 * torch.sum(1 + logvar-prior_logvar - (torch.abs(mu-prior_mu)-self.r).pow(2) - (logvar-prior_logvar).exp())
        return recon_bce, var_loss
            

    def forward(self, x):
        # encoding
        mu, log_var = self.encode(x)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

