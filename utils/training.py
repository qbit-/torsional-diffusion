import numpy as np
from tqdm import tqdm
import torch
import diffusion.torus as torus


def train_epoch(accelerator, model, loader, optimizer):
    model.train()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        
        data.to(accelerator.device)
        data = model(data)
        pred = data.edge_pred

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score).to(accelerator.device)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm).to(accelerator.device)
        loss = ((score - pred) ** 2 / score_norm).mean()
        base_noise = (score ** 2 / score_norm).mean()

        accelerator.backward(loss)
        optimizer.step()
        
        loss_batch, base_noise_batch = accelerator.gather_for_metrics((loss, base_noise))
        
        loss_tot += loss_batch.mean().item()
        base_tot += base_noise_batch.mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg


@torch.no_grad()
def test_epoch(accelerator, model, loader):
    model.eval()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):

        data = data.to(accelerator.device)
        data = model(data)
        pred = data.edge_pred.cpu()

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score).to(accelerator.device)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm).to(accelerator.device)
        loss = ((score - pred) ** 2 / score_norm).mean()
        base_noise = (score ** 2 / score_norm).mean()

        loss_batch, base_noise_batch = accelerator.gather_for_metrics((loss, base_noise))

        loss_tot += loss_batch.mean().item()
        base_tot += base_noise_batch.mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg

