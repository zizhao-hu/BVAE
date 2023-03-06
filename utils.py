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
    imageio.mimsave(cwd + '/outputs_2/generated_images.gif', imgs)

def save_reconstructed_images(recon_images, epoch, name):
    save_image(recon_images.cpu(), cwd + f"/outputs_2/{name}_output{epoch}.jpg")

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
    plt.savefig(cwd +'/outputs_2/' + f"{ylabel}.jpg")
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
            scatter = plt.subplot(3,3,3*i+j).scatter(latent[:,j], latent[:,i], c=y, cmap = 'tab10', s =1)
            if j == i+1:
                plt.xlabel(f"Latent {j+1}")
                plt.ylabel(f"Latent {i+1}")
            plt.axis('square')
            plt.axis('equal')
    plt.figlegend(*scatter.legend_elements(), loc = 'lower left', ncol=5, labelspacing=0.1, prop={'size': 8})
    plt.savefig(cwd +f'/outputs_2/{model.name}_latent.jpg')
    plt.show()


def save_inter_latent(batch_img, model):
    latentt,_ = model.encode(batch_img)
    latentb,_ = model.encode(batch_img)
    latent_max,_ = torch.max(latentt, 0)
    latent_min,_ = torch.min(latentt, 0)
    idxb = torch.topk(latent_max-latent_min, 2,largest = False).indices
    idxt = torch.topk(latent_max-latent_min, 2).indices
    
    for i in range(8):
        for j in range(8):
            latentt[i*8+j] = latentt[0]
            latentt[i*8+j][idxt[0]] = latent_min[idxt[0]] + i/7*(latent_max[idxt[0]]-latent_min[idxt[0]])
            latentt[i*8+j][idxt[1]] = latent_min[idxt[1]] + j/7*(latent_max[idxt[1]]-latent_min[idxt[1]])

            latentb[i*8+j] = latentb[0]
            latentb[i*8+j][idxb[0]] = latent_min[idxb[0]] + i/7*(latent_max[idxb[0]]-latent_min[idxb[0]])
            latentb[i*8+j][idxb[1]] = latent_min[idxb[1]] + j/7*(latent_max[idxb[1]]-latent_min[idxb[1]])
    imgt = model.decode(latentt)
    imgb = model.decode(latentb)
    save_image(imgt.cpu(), cwd + f"/outputs_2/{model.name}_iter_t2.jpg")
    save_image(imgb.cpu(), cwd + f"/outputs_2/{model.name}_iter_b2.jpg")

def le_score(weight):
    vec = nn.functional.normalize(weight, p=2, dim=1)
    vec = torch.pow(torch.mm(vec,vec.T),2)
    le = (torch.sum(vec)-vec.shape[0])/vec.shape[0]/(vec.shape[0]-1)
    return le.item()