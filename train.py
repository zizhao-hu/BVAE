import torch
import torch.optim as optim
import torch.nn as nn
from model.ConvVAE import ConvVAE
import torchvision.transforms as transforms
import torchvision
import matplotlib

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import engine
from utils import save_reconstructed_images, image_to_vid, save_loss_plot, save_latent_scatter, le_score
import os
path = os.getcwd()

matplotlib.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = ConvVAE().to(device)

# set the learning parameters
lr = 0.001
epochs = 100
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)

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

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = engine.train(
        model, trainloader, trainset, device, optimizer,
    )
    valid_epoch_loss, recon_images = engine.validate(
        model, testloader, testset, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

mus, ys = engine.latent(
        model, testloader, testset, device
    )

    # save the reconstructions as a .gif file
image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
save_latent_scatter(mus, ys)
print("le_score:",le_score(model.fc_mu.weight.data))
print('TRAINING COMPLETE')