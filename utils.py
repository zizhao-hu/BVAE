import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

to_pil_image = transforms.ToPILImage()
cwd = os.getcwd()

def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(cwd + '/outputs/generated_images.gif', imgs)

def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), cwd + f"/outputs/output{epoch}.jpg")

def save_plot(dict, xlabel='x', ylabel='y'):
    # loss plots
    plt.figure(figsize=(10, 7))
    for model_name, metrics in dict.items():
        for key, val in metrics.items():
            if key == ylabel:
                plt.plot(val,label=model_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(cwd +'/outputs/' + f"{ylabel}.jpg")
    plt.show()

def latent(model, dataloader, dataset, device):
    model.eval()
    with torch.no_grad():
        for i, (data, y) in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            data = data.to(device)
            mu, _ = model.encode(data)
            if i == 0:
                mus = mu
                ys = y
            else:
                mus = torch.concatenate((mus,mu))
                ys = torch.concatenate((ys,y))
    return mus, ys

def save_latent_scatter(model, dataloader, dataset, device):
    latent, y = latent(model, dataloader, dataset, device)
    latent = latent.detach().cpu().numpy()
    y = y.cpu().numpy()
    plt.figure(figsize=(10,10))
    for i in range(4):
        for j in range(i,4):
    
            scatter = plt.subplot(4,4,4*i+j+1).scatter(latent[:,j], latent[:,i], c=y, cmap = 'tab10', s =3)
            if j==i:
                plt.xlabel(f"Latent {j+1}")
                plt.ylabel(f"Latent {i+1}")
            plt.axis('square')
    
    plt.figlegend(*scatter.legend_elements(), loc = 'lower center', ncol=5, labelspacing=0.)
    plt.savefig(cwd +f'/outputs/{model.name}_latent.jpg')
    plt.show()


def le_score(weight):
    vec = nn.functional.normalize(weight, p=2, dim=1)
    vec = torch.pow(torch.mm(vec,vec.T),2)
    return torch.sum(vec)/vec.shape[0]/vec.shape[1]