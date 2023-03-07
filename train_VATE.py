import torch
import torch.optim as optim
import torch.nn as nn
from model.VATE import VATE
from model.ConvVAE import ConvVAE

import torchvision.transforms as transforms
import torchvision
import matplotlib as mpl
from collections import defaultdict
tree = lambda: defaultdict(tree)
from torch.utils.data import DataLoader
from torchvision.utils import make_grid,save_image
import matplotlib.pyplot as plt
import engine
import numpy as np
from cycler import cycler
from utils.vis import save_reconstructed_images, image_to_vid, save_plot, save_latent_scatter, save_inter_latent,plot_gaussian
from utils.math import le_score
from utils.utils import get_latent
import os
path = os.getcwd()
from umap import UMAP
cwd = os.getcwd()
mpl.style.use('ggplot')

color = plt.cm.tab20(np.linspace(0, 1,20))
mpl.rcParams['axes.prop_cycle'] = cycler('color', color)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
amodel = VATE(name = "VATE_V").to(device)
bmodel = VATE(name = "VATE_N", norm = True).to(device)
cmodel = VATE(beta = 10, name = "VATE_NB", norm = True).to(device)
dmodel = ConvVAE().to(device)
emodel = ConvVAE(name = "AE").to(device)

# set the learning parameters
lr = 0.001
epochs = 40
batch_size = 64

aoptimizer = optim.Adam(amodel.parameters(), lr=lr)
boptimizer = optim.Adam(bmodel.parameters(), lr=lr)
coptimizer = optim.Adam(cmodel.parameters(), lr=lr)
doptimizer = optim.Adam(dmodel.parameters(), lr=lr)
eoptimizer = optim.Adam(emodel.parameters(), lr=lr)


# a list to save all the reconstructed images in PyTorch grid format


# ###### MNIST ######
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
# ])
# # training set and train data loader
# trainset = torchvision.datasets.MNIST(
#     root=path +'/data', train=True, download=True, transform=transform
# )
# trainloader = DataLoader(
#     trainset, batch_size=batch_size, shuffle=True
# )
# # validation set and validation data loader
# testset = torchvision.datasets.MNIST(
#     root=path +'/data', train=False, download=True, transform=transform
# )
# testloader = DataLoader(
#     testset, batch_size=batch_size, shuffle=False
# )
# ###### MNIST ######

###### CELEBA ######

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
# training set and train data loader
celebaset = torchvision.datasets.ImageFolder(
    root=path +'/data/celeba',  transform=transform
)

trainset = torch.utils.data.Subset(celebaset, list(range(0,10)))

trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

testset = torch.utils.data.Subset(celebaset, list(range(60000,65000)))

testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

###### CELEBA ######



##### experiment 1 ######
dict = defaultdict(lambda: defaultdict(list))
models = [amodel, bmodel, cmodel,dmodel,emodel]
optimizers = [aoptimizer, boptimizer, coptimizer, doptimizer, eoptimizer]

for i, model in enumerate(models):
    grid_images = []
    optimizer = optimizers[i]
    for epoch in range(epochs):
        print(f"{model.name}: Epoch {epoch+1} of {epochs}")
        train_epoch_loss = engine.train(
            model, trainloader, trainset, device, optimizer,
        )
        
        for i in range(16):
            plt.axvline(i*3, color='grey',linestyle = '--')
            plot = plot_gaussian(i, var[i].detach().cpu(),legend_label = 'posterior', color = 'blue',linewidth =2)
            plot = plot_gaussian(i, model.curlogvar[i].detach().cpu(),legend_label = 'prior', color = 'red',linewidth =2)
            plt.xticks([]) 
        plt.gca().set_xlim(-5,60)
        plt.savefig(cwd +f'/outputs/gaussian.jpg')
        valid_epoch_recon_loss,valid_epoch_elbo, recon_images = engine.validate(
            model, testloader, testset, device
        )
        myle_score = le_score(model.fc_mu.weight.data)
        dict[model.name]["train_loss"].append(train_epoch_loss)
        dict[model.name]["valid_elbo"].append(valid_epoch_elbo)
        dict[model.name]["valid_recon_loss"].append(valid_epoch_recon_loss)
        dict[model.name]["le_score"].append(myle_score)

        # # save the reconstructed images from the validation loop
        if i ==0 and epoch == 0:
            save_reconstructed_images(iter(testloader).__next__()[0], epoch, model.name)
        save_reconstructed_images(recon_images, epoch+1, model.name)
        # convert the reconstructed images to PyTorch image grid format
        image_grid = make_grid(recon_images.detach().cpu())
        grid_images.append(image_grid)
    image_to_vid(grid_images)
    save_inter_latent(iter(testloader).__next__()[0].to(device), model)
        # print(f"Train Loss: {train_epoch_loss:.4f}")
        # print(f"Val Loss: {valid_epoch_loss:.4f}")


#     # save the reconstructions as a .gif file
# # save the loss plots to disk
save_plot(dict,xlabel = "Epochs",ylabel ="valid_recon_loss")
save_plot(dict,xlabel = "Epochs",ylabel ="le_score")
save_plot(dict,xlabel = "Epochs",ylabel ="valid_elbo")
save_latent_scatter(amodel, testloader, testset, device)
save_latent_scatter(bmodel, testloader, testset, device)
save_latent_scatter(cmodel, testloader, testset, device)
save_latent_scatter(dmodel, testloader, testset, device)
save_latent_scatter(emodel, testloader, testset, device)

##### experiment 2 ######

# dict = defaultdict(lambda: defaultdict(list))
# r = [0,0.1,1,2]
# beta = [1, 10, 10]
# C = [0, 0, 10]
# plt.figure(figsize=(10,8))
# for i_row in range(3):
#     for i_col in range(4):
#         myr = r[i_col]
#         myb = beta[i_row]
#         myC = C[i_row]
#         model = ConvVAE(name = f"r={myr}_b={myb}_C={myC}_ConvVAE",r = myr, beta =myb, C = myC).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         for epoch in range(epochs):
#             print(f"{model.name}: Epoch {epoch+1} of {epochs}")
#             train_epoch_loss = engine.train(
#                 model, trainloader, trainset, device, optimizer,
#             )
#             valid_epoch_recon_loss,valid_epoch_elbo, recon_images = engine.validate(
#                 model, testloader, testset, device
#             )
#             myle_score = le_score(model.fc_mu.weight.data)
#             dict[model.name]["train_loss"].append(train_epoch_loss)
#             dict[model.name]["valid_elbo"].append(valid_epoch_elbo)
#             dict[model.name]["valid_recon_loss"].append(valid_epoch_recon_loss)
#             dict[model.name]["le_score"].append(myle_score)
        
#         latent, y = get_latent(model, testloader, testset, device)
#         latent = latent.detach().cpu().numpy()
#         y = y.cpu().numpy()
#         latent = UMAP().fit_transform(latent)
        
#         scatter = plt.subplot(3,4,4*i_row+i_col+1).scatter(latent[:,0], latent[:,1], c=y, cmap='tab10',s =0.5)
        
# plt.figlegend(*scatter.legend_elements(), loc = 'lower left', ncol=5, labelspacing=0.1)
# plt.savefig(cwd +f'/outputs/all_umap.jpg')
# plt.show()

# save_plot(dict,xlabel = "Epochs",ylabel ="valid_recon_loss")
# save_plot(dict,xlabel = "Epochs",ylabel ="le_score")
# save_plot(dict,xlabel = "Epochs",ylabel ="valid_elbo")

# print('TRAINING COMPLETE')