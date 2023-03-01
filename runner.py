
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
        reconstruction, mu, log_var = model(data)
        bce_loss, var_loss = model.loss(data, reconstruction, mu, log_var)
        print(bce_loss.item(), var_loss.item())
        loss = bce_loss + var_loss
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        
    train_loss = running_loss / counter 
    return train_loss


def validate(model, dataloader, dataset, device):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, log_var = model(data)
            bce_loss, var_loss = model.loss(data, reconstruction, mu, log_var)
            loss = bce_loss + var_loss
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images