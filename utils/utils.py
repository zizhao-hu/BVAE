import torch
from tqdm import tqdm
import os

cwd = os.getcwd()

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



