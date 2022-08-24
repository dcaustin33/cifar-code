#gneeric python imports
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb

#custom imports
from utils import  top_5_error_rate_metric, top_1_error_rate_metric
import cifar_100
from logger import log_metrics as logger
import flax_trainer as trainer
from flax.training import train_state
from cifar_resnet import resnet18
import cifar_100

#python helper inputs
import os
import time


def prepare_data(dataset_args, val_dataset_args):
    
    train_dataset = cifar_100.CIFAR_100_transformations(train = True, **dataset_args)
    dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else 0,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    val_dataset = cifar_100.CIFAR_100_transformations(views = 1, train = True, **val_dataset_args)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        num_workers=args.workers if args.workers else 0,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False
    )
    return train_dataset, dataloader, val_dataloader


def create_params(model, example, rng):
    model = model
    batch = example  # (N, H, W, C) format
    params = model.init(rng, batch)['params']
    return params


def create_optimizer(optimizer, lr, weight_decay, params):
    optimizer = optimizer(lr, weight_decay = weight_decay)
    optimizer_state = optimizer.init(params)
    return optimizer, optimizer_state


def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

@jax.jit
def training_step(images, 
                  labels,
                   params):

    def loss_fn(params):
        logits, _ = model.apply({'params': params}, images, mutable=['batch_stats'])
        loss = cross_entropy_loss(logits=logits, labels=labels)
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    acc1 = top_1_error_rate_metric(logits = logits, one_hot_labels = labels) 
    acc5 = top_5_error_rate_metric(logits = logits, one_hot_labels = labels) 
    
    return grads, acc1, acc5, loss

@jax.jit
def validation_step(images, 
                  labels,
                   params):

    logits, _ = model.apply({'params': params}, images, mutable=['batch_stats'])
    acc1 = top_1_error_rate_metric(logits = logits, one_hot_labels = labels) 
    acc5 = top_5_error_rate_metric(logits = logits, one_hot_labels = labels) 
    
    return grads, acc1, acc5, loss




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CIFAR 100 Test Runs')
    parser.add_argument('--name', default = 'CIFAR_100_Supervised', type = str)
    parser.add_argument('--lr', nargs='?', default = .001, type=float)
    parser.add_argument('--workers', nargs='?', default = 8,  type=int)
    parser.add_argument('--steps', nargs='?', default = 10000,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 128,  type=int)
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
    
    #create the model and the optimizer
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    model = resnet18()
    params = create_params(model, jnp.ones((4, 32, 32, 3)), rng)
    schedule1 = optax.linear_schedule(3e-05, args.lr, args.warmup_steps)
    schedule2 = optax.cosine_decay_schedule(args.lr, args.steps)
    schedule = optax.join_schedules([schedule1, schedule2], [args.warmup_steps])

    schedule = optax.warmup_cosine_decay_schedule(
                                                    init_value=3e-05,
                                                    peak_value=args.lr,
                                                    warmup_steps=args.warmup_steps,
                                                    decay_steps=args.steps,
                                                    end_value=args.lr * .01,
                                                    )


    optimizer = optax.adamw
    optimizer, optimizer_state = create_optimizer(optimizer, schedule, args.wd, params)
    
    
    #prepare the data
    dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)

    #log if applicable
    if args.log and args.rank == 0:
        wandb = wandb.init(config = args, name = args.name, project = 'CIFAR')
    else: wandb = None
    steps = 0
        
    #instantiate and train
    trainer = trainer.Trainer(
                             model = model,
                             params = params,
                             dataloader = dataloader, 
                             val_dataloader = val_dataloader,
                             args = args, 
                             training_step = training_step,
                             validation_step = validation_step,
                             optimizer = optimizer, 
                             optimizer_state = optimizer_state, 
                             current_step = steps,
                             metrics = metrics,
                             val_metrics = val_metrics,
                             wandb = wandb)
    
    trainer.train()
    
    
    
    
    
    