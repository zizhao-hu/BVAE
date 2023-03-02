import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
import os
from cycler import cycler
import matplotlib as mpl
color = plt.cm.tab20(np.linspace(0, 1,20))
mpl.rcParams['axes.prop_cycle'] = cycler('color', color)

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
    plt.legend(loc = 'lower left')
    plt.savefig(cwd +'/outputs/' + f"{ylabel}.jpg")
    plt.show()

def get_latent(model, dataloader, dataset, device):
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
    latent, y = get_latent(model, dataloader, dataset, device)
    latent = latent.detach().cpu().numpy()
    y = y.cpu().numpy()
    plt.figure(figsize=(8,8))
    for i in range(4):
        for j in range(i,4):
            if j==i:
                continue
            scatter = plt.subplot(3,3,3*i+j).scatter(latent[:,j], latent[:,i], c=y, cmap = 'tab10', s =2)
            if j == i+1:
                plt.xlabel(f"Latent {j+1}")
                plt.ylabel(f"Latent {i+1}")
            plt.axis('square')
            plt.axis('equal')
    plt.figlegend(*scatter.legend_elements(), loc = 'lower left', ncol=5, labelspacing=0.1, prop={'size': 8})
    plt.savefig(cwd +f'/outputs/{model.name}_latent.jpg')
    plt.show()

    
def le_score(weight):
    vec = nn.functional.normalize(weight, p=2, dim=1)
    vec = torch.pow(torch.mm(vec,vec.T),2)
    le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
    return le.item()