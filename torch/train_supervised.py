import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from torch.multiprocessing import Pool, Process, set_start_method
import torch.distributed as dist

#imports defined in folder
from utils import get_accuracy
from lars import LARSWrapper
import cifar_100_data
from logger import log_metrics as logger
import torch_cnn
import torch_trainer as trainer

#python helper inputs
import os
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
import pytorch_lightning as pl
import time


def prepare_data(dataset_args, val_dataset_args):
    
    train_dataset = cifar_100_data.CIFAR_100_transformations(train = True, **dataset_args)
    dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else len(os.sched_getaffinity(0)),
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    val_dataset = cifar_100_data.CIFAR_100_transformations(views = 1, train = True, **val_dataset_args)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else len(os.sched_getaffinity(0)),
        pin_memory=False,
        drop_last=True,
        persistent_workers=False
    )
    return train_dataset, dataloader, val_dataloader


def training_step(data: list, 
               model: nn.Module, 
               metrics: dict,
               step: int,
               log = False,
               wandb = None,
               args = None) -> torch.Tensor:
 
    labels = data['label'].cuda()
    classification1 = model(data['image0'].cuda())

    loss1 = F.cross_entropy(classification1, labels)
    acc1, acc5 = get_accuracy(classification1, labels)
    
    metrics['total'] += data['image0'].size(0)
    metrics['Accuracy'] += acc1
    metrics['Accuracy Top 5'] += acc5
    
    metrics['Loss'] += loss1
    
    
    #logging protocol
    if log:# and args.rank == 0:
        logger(metrics, step, wandb = wandb, train = True)
    
    return loss1

    
def validation_step(data: list, 
                    model: nn.Module, 
                    metrics: dict,
                    step: int,
                    log = False,
                    wandb = None,
                    args = None):
    with torch.no_grad():
        labels = data['label'].cuda()
        classification1 = model(data['image0'].cuda())
        acc1, acc5 = get_accuracy(classification1, labels)

        metrics['total'] += data['image0'].size(0)
        metrics['Accuracy'] += acc1
        metrics['Accuracy Top 5'] += acc5


        #logging protocol
        if log:# and args.rank == 0:
            logger(metrics, step, wandb = wandb, train = False)

        return 




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CIFAR 100 Test Runs')
    parser.add_argument('--name', default = 'CIFAR_100_Supervised', type = str)
    parser.add_argument('--lr', nargs='?', default = .001, type=float)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--steps', nargs='?', default = 10000,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 256,  type=int)
    parser.add_argument('--val_steps', nargs='?', default = 70,  type=int)
    parser.add_argument('--log_n_steps', nargs='?', default = 800,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = 'CIFAR-100', type = str)
    parser.add_argument('--gpus', default = 1, type = int)
    parser.add_argument('--log_n_train_steps', default = 100, type = int)
    parser.add_argument('-checkpoint', action='store_true')
    parser.add_argument('--checkpoint_path', default = None, type = str)
    
    #distributed arguments
    parser.add_argument("--dist_url", default="tcp://localhost:40000", type=str,
                    help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, 
                    help="""number of processes: it is set automatically and should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, 
                    help="""rank of this process: it is set automatically and should not be passed as argument""")
    parser.add_argument("--job_id", default=0, type=int, 
                    help="""rank of this process: it is set automatically and should not be passed as argument""")
    args = parser.parse_args()
    if args.data_path[-1] != '/': args.data_path = args.data_path + '/'
        
    args.wd = 1.5e-6
    args.steps += 1
    args.warmup_steps = 200
    args.log_n_val_steps = args.val_steps - 1
        
    args.dataset_args = {
                 'crop_size': 32,
                 'brightness': 0.4, 
                 'contrast': 0.4, 
                 'saturation': .2, 
                 'hue': .1, 
                 'color_jitter_prob': .6, 
                 'gray_scale_prob': 0.2, 
                 'horizontal_flip_prob': 0.5, 
                 'gaussian_prob': .5, 
                 'min_scale': 0.4, 
                 'max_scale': 0.9}
    
    args.val_dataset_args = {
                 'crop_size': 32,
                 'brightness': 0.4, 
                 'contrast': 0.4, 
                 'saturation': .2, 
                 'hue': .1, 
                 'color_jitter_prob': 0, 
                 'gray_scale_prob': 0, 
                 'horizontal_flip_prob': 0.5, 
                 'gaussian_prob': 0, 
                 'min_scale': 0.9, 
                 'max_scale': 1}
    
    #set_distributed_mode(args)
    

    metrics = {}
    metrics['total'] = 0
    metrics['Loss'] = 0
    metrics['Accuracy'] = 0
    metrics['Accuracy Top 5'] = 0


    val_metrics = {}
    val_metrics['total'] = 0
    val_metrics['CE Loss'] = 0
    val_metrics['Accuracy'] = 0
    val_metrics['Accuracy Top 5'] = 0
    
    model = torch_cnn.res18().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    #optimizer = LARSWrapper(optimizer, clip = True, exclude_bias_n_norm = True, eta = .001)
    schedule = LinearWarmupCosineAnnealingLR(
                            optimizer,
                            warmup_epochs= args.warmup_steps,
                            max_epochs= args.steps,
                            warmup_start_lr=3e-05,
                            eta_min=.0001)
    
    
    dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)
    if args.log and args.rank == 0:
        wandb = wandb.init(config = args, name = args.name, project = 'CIFAR')
    else: wandb = None
    steps = 0
        
    trainer = trainer.Trainer(
                             model,
                             dataloader, 
                             val_dataloader,
                             args, 
                             training_step,
                             validation_step,
                             optimizer, 
                             schedule, 
                             current_step = steps,
                             metrics = metrics,
                             val_metrics = val_metrics,
                             wandb = wandb)
    
    trainer.train()
    
    
    
    
    
    