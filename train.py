import math, os, torch, yaml
torch.multiprocessing.set_sharing_strategy('file_descriptor')
from accelerate import Accelerator, DistributedDataParallelKwargs
import numpy as np
from rdkit import RDLogger
from utils.dataset import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch
from utils.utils import get_model, get_optimizer_and_scheduler, save_yaml_file
from utils.boltzmann import BoltzmannResampler
from argparse import Namespace

RDLogger.DisableLog('rdApp.*')

"""
    Training procedures for both conformer generation and Botzmann generators
    The hyperparameters are taken from utils/parsing.py and can be given as arguments
"""


def train(args, accelerator, model, optimizer, scheduler, train_loader, val_loader):
    best_val_loss = math.inf
    best_epoch = 0

    accelerator.print("Starting training...")
    for epoch in range(args.n_epochs):

        train_loss, base_train_loss = train_epoch(accelerator, model, train_loader, optimizer)
        accelerator.print("Epoch {}: Training Loss {}  base loss {}".format(epoch, train_loss, base_train_loss))

        val_loss, base_val_loss = test_epoch(accelerator, model, val_loader)
        accelerator.print("Epoch {}: Validation Loss {} base loss {}".format(epoch, val_loss, base_val_loss))

        if scheduler:
            scheduler.step(val_loss)

        if accelerator.is_main_process:
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                accelerator.save(
                    accelerator.unwrap_model(model).state_dict(),
                    os.path.join(args.log_dir, 'best_model.pt'))

            accelerator.save({
                'epoch': epoch,
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(args.log_dir, 'last_model.pt'))

    accelerator.print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))


def boltzmann_train(args, accelerator, model, optimizer, train_loader, val_loader, resampler):
    accelerator.print("Starting training...")

    val_ess = val_loader.dataset.resample_all(resampler, temperature=args.temp)
    accelerator.print(f"Initial val ESS: Mean {np.mean(val_ess):.4f} Median {np.median(val_ess):.4f}")
    best_val = val_ess

    for epoch in range(args.n_epochs):
        if args.adjust_temp:
            train_loader.dataset.boltzmann_resampler.temp = (3000 - args.temp) / (epoch + 1) + args.temp

        train_loss, base_train_loss = train_epoch(accelerator, model, train_loader, optimizer)
        accelerator.print("Epoch {}: Training Loss {}  base loss {}".format(epoch, train_loss, base_train_loss))
        if epoch % 5 == 0:
            val_ess = val_loader.dataset.resample_all(resampler, temperature=args.temp)
            accelerator.print(f"Epoch {epoch} val ESS: Mean {np.mean(val_ess):.4f} Median {np.median(val_ess):.4f}")

            if accelerator.is_main_process:
                if best_val > val_ess:
                    best_val = val_ess
                    accelerator.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(args.log_dir, 'best_model.pt'))

                accelerator.save({
                    'epoch': epoch,
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(args.log_dir, 'last_model.pt'))


if __name__ == '__main__':
    args = parse_train_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # build model
    if args.restart_dir:
        with open(f'{args.restart_dir}/model_parameters.yml') as f:
            args_old = Namespace(**yaml.full_load(f))

        model = get_model(args_old).to(accelerator.device)
        state_dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)

    else:
        model = get_model(args).to(accelerator.device)

    numel = sum([p.numel() for p in model.parameters()])

    # construct loader
    if args.boltzmann_training:
        boltzmann_resampler = BoltzmannResampler(args, model)
    else:
        boltzmann_resampler = None
    train_loader, val_loader = construct_loader(args, boltzmann_resampler=boltzmann_resampler)

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    # prepare objects with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    # record parameters
    yaml_file_name = os.path.join(args.log_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)

    if args.boltzmann_training:
        boltzmann_train(args, accelerator, model, optimizer, train_loader, val_loader, boltzmann_resampler)
    else:
        train(args, accelerator, model, optimizer, scheduler, train_loader, val_loader)
