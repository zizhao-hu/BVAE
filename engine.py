
import torch; torch.manual_seed(0)
from tqdm import tqdm

def train(model, dataloader, dataset, device, optimizer):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, est_mu, est_logvar = model(data)
        
        agg_mu = torch.mean(est_mu, dim = 0)
        agg_logvar =  torch.log(torch.var(est_mu, dim = 0))

        prior_mu = (agg_mu + est_mu)/2
        prior_var = (agg_logvar + est_logvar)/2

        bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_mu, prior_mu.detach(), prior_var.detach())
        if model.C != 0:
            var_loss = abs(var_loss - model.C)
        if model.beta != 1:
            var_loss = var_loss * model.beta
        loss = bce_loss + var_loss
        
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
            reconstruction, est_mu, est_logvar = model(data)
            agg_mu = torch.mean(est_mu, dim = 0)
            agg_logvar =  torch.log(torch.var(est_mu, dim = 0))

            prior_mu = (agg_mu + est_mu)/2
            prior_var = (agg_logvar + est_logvar)/2

            bce_loss, var_loss = model.loss(data, reconstruction, est_mu, est_mu, prior_mu.detach(), prior_var.detach())
            loss = bce_loss + var_loss
            running_bce_loss += bce_loss.item()/dataloader.batch_size
            running_elbo += -loss.item()/dataloader.batch_size

            # save the last batch input and output of every epoch
            if i == 0:
                recon_images = reconstruction
    val_recon_loss = running_bce_loss / counter
    val_elbo = running_elbo / counter
    return val_recon_loss, val_elbo, recon_images

