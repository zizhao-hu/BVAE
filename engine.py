
import torch; torch.manual_seed(0)
from tqdm import tqdm
from utils.math import kl
def train(model, dataloader, dataset, device, optimizer):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        if 'VATE' in model.name:
            reconstruction, est_mu, est_logvar,prior_mu, prior_logvar = model(data)
            bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_logvar, prior_mu.detach(), prior_logvar.detach())
        elif 'FVAE' in model.name:
            reconstruction, est_mu, est_logvar,prior_mu, prior_logvar = model(data)
            bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_logvar, prior_mu.detach(), prior_logvar.detach())
            model.eval()
            with torch.no_grad():
                recon_mu, recon_logvar = model.encode(reconstruction)
                bce_loss += kl(est_mu, est_logvar, recon_mu, recon_logvar)
        elif 'HBVAE' in model.name:
            print("HBVAE")
            reconstruction, prior_mu, agg_mu = model(data)
            bce_loss, var_loss = model.loss(data, reconstruction, agg_mu, prior_mu=prior_mu)
        elif "IVAE" in model.name:
            print("IVAE")
            xs, en_recons, de_recons, mus, logvars = model(data)
            bce_loss, var_loss = model.loss(xs, en_recons, de_recons, mus, vars)
        else:
            reconstruction, est_mu, est_logvar = model(data)
            bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_logvar)
        
        if model.C != 0:
            var_loss = abs(var_loss - model.C)
        if model.beta != 1:
            var_loss = var_loss * model.beta
        loss = bce_loss + var_loss
        if model.norm:
            loss += model.le_score()
        
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        
    train_loss = running_loss / counter 
    return train_loss


def validate(model, dataloader, dataset, device):
    model.eval()
    running_bce_loss = 0.0
    running_elbo = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            if 'VATE' in model.name:
                reconstruction, est_mu, est_logvar,prior_mu, prior_logvar = model(data)
                bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_logvar, prior_mu.detach(), prior_logvar.detach())
            elif 'FVAE' in model.name:
                reconstruction, est_mu, est_logvar,prior_mu, prior_logvar = model(data)
                bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_logvar, prior_mu.detach(), prior_logvar.detach())
            else:
                reconstruction, est_mu, est_logvar = model(data)
                bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_logvar)

            loss = bce_loss + var_loss
            running_bce_loss += bce_loss.item()/dataloader.batch_size
            running_elbo += -loss.item()/dataloader.batch_size

            # save the last batch input and output of every epoch
            if i == 0:
                recon_images = reconstruction
    val_recon_loss = running_bce_loss / counter
    val_elbo = running_elbo / counter
    return val_recon_loss, val_elbo, recon_images

