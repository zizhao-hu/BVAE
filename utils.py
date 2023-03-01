import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

to_pil_image = transforms.ToPILImage()
path = os.getcwd()

def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(path + '/outputs/generated_images.gif', imgs)

def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), path + f"/outputs/output{epoch}.jpg")

def save_loss_plot(train_loss, valid_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path +'/outputs/loss.jpg')
    plt.show()

def save_latent_scatter(latent, y):
    latent = latent.detach().cpu().numpy()
    y = y.cpu().numpy()
    plt.figure(figsize=(10,10))
    for i in range(4):
        for j in range(i,4):
    
            scatter = plt.subplot(4,4,4*i+j+1).scatter(latent[:,i], latent[:,j], c=y, cmap = 'tab10', s =5)

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    legend = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    plt.savefig(path +'/outputs/nonbilatent.jpg')
    plt.show()