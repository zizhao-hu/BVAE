import torch
import torch.optim as optim
import torch.nn as nn
from model.ConvVAE import ConvVAE
import torchvision.transforms as transforms
import torchvision
import matplotlib
from collections import defaultdict
tree = lambda: defaultdict(tree)
from torch.utils.data import DataLoader
from torchvision.utils import make_grid,save_image
import matplotlib.pyplot as plt
import engine
from utils import save_reconstructed_images, image_to_vid, save_plot, save_latent_scatter, le_score,get_latent
import os
path = os.getcwd()
from umap import UMAP
cwd = os.getcwd()
matplotlib.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
amodel = ConvVAE().to(device)
bmodel = ConvVAE(r=1, name = "Binarized-r1").to(device)
cmodel = ConvVAE(beta=10, name = "Beta-b10").to(device)
dmodel = ConvVAE(beta = 10, C=20, name = "DBeta-b10-C20").to(device)
emodel = ConvVAE(beta = 10, C=20, r=0.5,name = "BDBeta-b10-C20-r1").to(device)
# set the learning parameters
lr = 0.001
epochs = 1
batch_size = 64

aoptimizer = optim.Adam(amodel.parameters(), lr=lr)
boptimizer = optim.Adam(bmodel.parameters(), lr=lr)
coptimizer = optim.Adam(cmodel.parameters(), lr=lr)
doptimizer = optim.Adam(dmodel.parameters(), lr=lr)
eoptimizer = optim.Adam(emodel.parameters(), lr=lr)

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
# training set and train data loader
trainset = torchvision.datasets.MNIST(
    root=path +'/data', train=True, download=True, transform=transform
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
# validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root=path +'/data', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

###### experiment 1 ######
# dict = defaultdict(lambda: defaultdict(list))
# models = [amodel, bmodel, cmodel, dmodel, emodel]
# optimizers = [aoptimizer, boptimizer, coptimizer,doptimizer, eoptimizer]

# for i, model in enumerate(models):
#     optimizer = optimizers[i]
#     for epoch in range(epochs):
#         if model.C != 0:
#             model.C = epoch/2+0.01
#         print(f"{model.name}: Epoch {epoch+1} of {epochs}")
#         train_epoch_loss = engine.train(
#             model, trainloader, trainset, device, optimizer,
#         )
#         valid_epoch_recon_loss,valid_epoch_elbo, recon_images = engine.validate(
#             model, testloader, testset, device
#         )
#         myle_score = le_score(model.fc_mu.weight.data)
#         dict[model.name]["train_loss"].append(train_epoch_loss)
#         dict[model.name]["valid_elbo"].append(valid_epoch_elbo)
#         dict[model.name]["valid_recon_loss"].append(valid_epoch_recon_loss)
#         dict[model.name]["le_score"].append(myle_score)
       
#         # # save the reconstructed images from the validation loop
#         # save_reconstructed_images(recon_images, epoch+1)
#         # # convert the reconstructed images to PyTorch image grid format
#         # image_grid = make_grid(recon_images.detach().cpu())
#         # grid_images.append(image_grid)
#         # print(f"Train Loss: {train_epoch_loss:.4f}")
#         # print(f"Val Loss: {valid_epoch_loss:.4f}")


# #     # save the reconstructions as a .gif file
# # image_to_vid(grid_images)
# # # save the loss plots to disk
# save_plot(dict,xlabel = "Epochs",ylabel ="valid_recon_loss")
# save_plot(dict,xlabel = "Epochs",ylabel ="le_score")
# save_plot(dict,xlabel = "Epochs",ylabel ="valid_elbo")
# save_latent_scatter(amodel, testloader, testset, device)
# save_latent_scatter(bmodel, testloader, testset, device)
# save_latent_scatter(cmodel, testloader, testset, device)
# save_latent_scatter(dmodel, testloader, testset, device)
# save_latent_scatter(emodel, testloader, testset, device)

###### experiment 2 ######

dict = defaultdict(lambda: defaultdict(list))
r = [0,0.1,1,2]
beta = [1, 10, 10]
C = [0, 0, 20]
plt.figure(figsize=(10,8))
for i_row in range(3):
    for i_col in range(4):
        myr = r[i_col]
        myb = beta[i_row]
        myC = C[i_row]
        model = ConvVAE(name = f"r={myr}_b={myb}_C={myC}_ConvVAE",r = myr, beta =myb, C = myC).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            if model.C != 0:
                model.C = epoch/2+0.01
            print(f"{model.name}: Epoch {epoch+1} of {epochs}")
            train_epoch_loss = engine.train(
                model, trainloader, trainset, device, optimizer,
            )
            valid_epoch_recon_loss,valid_epoch_elbo, recon_images = engine.validate(
                model, testloader, testset, device
            )
            myle_score = le_score(model.fc_mu.weight.data)
            dict[model.name]["train_loss"].append(train_epoch_loss)
            dict[model.name]["valid_elbo"].append(valid_epoch_elbo)
            dict[model.name]["valid_recon_loss"].append(valid_epoch_recon_loss)
            dict[model.name]["le_score"].append(myle_score)
        
        latent, y = get_latent(model, testloader, testset, device)
        latent = latent.detach().cpu().numpy()
        y = y.cpu().numpy()
        latent = UMAP().fit_transform(latent)
        
        scatter = plt.subplot(3,4,3*i_row+i_col+1).scatter(latent[:,0], latent[:,1], c=y, cmap='tab10',s =2)

plt.figlegend(*scatter.legend_elements(), loc = 'lower left', ncol=5, labelspacing=0.1)
plt.savefig(cwd +f'/outputs/all_umap.jpg')
plt.show()

save_plot(dict,xlabel = "Epochs",ylabel ="valid_recon_loss")
save_plot(dict,xlabel = "Epochs",ylabel ="le_score")
save_plot(dict,xlabel = "Epochs",ylabel ="valid_elbo")

print('TRAINING COMPLETE')