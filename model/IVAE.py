import torch; torch.manual_seed(0)
from torch.nn import BCELoss
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d, Linear
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F

### model configs
kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 3 
latent_dim = 32 # latent dimension for sampling
###

class EncoderBlock(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))

class DecoderBlock(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = ConvTranspose2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = ConvTranspose2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))
	

class Latent(Module):
	def __init__(self, in_dim, latent_dim):
		super().__init__()
		# store the convolution and RELU layers
		self.fc_mu = Linear(in_dim, latent_dim)
		self.fc_out = Linear(latent_dim, in_dim)
		self.fc_logvar = Linear(in_dim, latent_dim)		
	
	def forward(self, x):	
		b, c, w, h = x.shape
		x = x.reshape(b, -1)
		mu = self.fc_mu(x)
		logvar = self.fc_logvar(x)
		z = self.reparameterize(mu, logvar)
		x = self.fc_out(z)
		x = x.reshape(b, c, w, h)
		return z, x, mu, logvar

class IVAE(Module):
	def __init__(self, in_dims, latent_dims, channels=(3, 16, 32, 64)):
		self.channels = channels
		self.encBlocks = ModuleList(
			[EncoderBlock(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.decBlocks = ModuleList(
			[DecoderBlock(channels[i], channels[i - 1])
			 	for i in range(1, len(channels))])
		self.latents = ModuleList(
			[Latent(in_dims[i], latent_dims[i])
    			for i in range(len(latent_dims))]
		)
		self.f1 = Linear(latent_dims[-1], 128)
		self.xs = []
		self.en_recon_xs = []
		self.de_recon_xs = []
		self.mus = []
		self.logvars = []

	def forward(self, x):
		## encode
		for i in range(len(self.channels) - 1):
			x1 = self.encBlocks[i](x)
			mu, logvar = self.latents[i](x1)
			
			en_recon_x = self.decBlocks[i](z) 
			self.xs.append(x)
			self.en_recon_xs.append(en_recon_x)
			self.mus.append(mu)
			self.logvars.append(logvar)
			x = x1
		
		x = self.f1(en_recon_x)
		self.de_recon_xs.insert(0, x)
		## decode
		for i in range(len(self.channels) - 3, -1, -1):
			x1 = self.decBlocks[i](x)			
			self.de_recon_xs.insert(0, x1)
			x = x1

		# return the final decoder output
		return self.xs, self.en_recon_xs, self.de_recon_xs, self.mus, self.logvars
	def loss(xs, en_recons, de_recons, mus, logvars):
		for i in len(en_recons):
			if i == 0:
				recon_bce = BCELoss(reduction='sum')(de_recons[0], xs[0])
				recon_bce += 2*BCELoss(reduction='sum')(en_recons[0], xs[0])
				recon_bce -= BCELoss(reduction='sum')(en_recons[0], de_recons[0])
				var_loss =  -0.5 * torch.sum(1 + logvars[0] - (mus[0]).pow(2) - (logvars[0]).exp())
			else:
				recon_bce += BCELoss(reduction='sum')(de_recons[i], xs[i])
				recon_bce += 2*BCELoss(reduction='sum')(en_recons[i], xs[i])
				recon_bce -= BCELoss(reduction='sum')(en_recons[i], de_recons[i])
				var_loss +=  -0.5 * torch.sum(1 + logvars[i] - (mus[i]).pow(2) - (logvars[i]).exp())
		return recon_bce, var_loss
		
	def reparameterize(self, mu, log_var):
		std = torch.exp(0.5*log_var) # standard deviation
		eps = torch.randn_like(std) # `randn_like` as we need the same size
		sample = mu + (eps * std) # sampling
		return sample