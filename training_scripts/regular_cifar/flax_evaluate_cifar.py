#gneeric python imports
import argparse
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import optax
import wandb

#custom imports
from utils import  top_1_error_rate_metric, top_5_error_rate_metric
import cifar_100_data
from flax.training import train_state, checkpoints
from flax_cifar_resnet import resnet18
import cifar_100_data
import flax_evaluator as evaluate

#python helper inputs
import time


def prepare_data(dataset_args, val_dataset_args):
    
    train_dataset = cifar_100_data.CIFAR_100_transformations(train = True, **dataset_args)
    dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=256,
        num_workers=8,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    val_dataset = cifar_100_data.CIFAR_100_transformations(views = 1, train = True, **val_dataset_args)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = True,
        batch_size=256,
        num_workers=8,
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


def create_optimizer(optimizer, lr, wd):
    optimizer = optimizer(lr)
    return optimizer


def create_train_state(rng, optimizer):
    """Creates initial `TrainState`."""
    model = resnet18()
    batch = jnp.ones((4, 32, 32, 3))  # (N, H, W, C) format
    params = model.init(jax.random.PRNGKey(0), batch)['params']
    tx = optimizer
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def master_val_step(params, data, metrics):
    logits = validation_step(params, data)
    acc1 = top_1_error_rate_metric(logits = logits, one_hot_labels = data['label']) 
    acc5 = top_5_error_rate_metric(logits = logits, one_hot_labels = data['label']) 

    metrics['total'] += data['image0'].shape[0]
    metrics['Accuracy'] += acc1
    metrics['Accuracy Top 5'] += acc5

    return None


@jax.jit
def validation_step(params, data):
    
    logits, _ = resnet18().apply({'params': params}, data['image0'], mutable=['batch_stats'])
    return logits




if __name__ == '__main__':

    dataset_args = {
                 'crop_size': 32,
                 'brightness': 0.4, 
                 'contrast': 0.4, 
                 'saturation': .2, 
                 'hue': .1, 
                 'color_jitter_prob': .8, 
                 'gray_scale_prob': 0.2, 
                 'horizontal_flip_prob': 0.5, 
                 'gaussian_prob': .5, 
                 'min_scale': 0.16, 
                 'max_scale': 0.9}
    val_dataset_args = {
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

    parser = argparse.ArgumentParser(description='CIFAR 100 Test Runs')
    parser.add_argument('--name', default = 'Eval Flax_CIFAR_100_Supervised', type = str)
    parser.add_argument('--workers', nargs='?', default = 1,  type=int)
    parser.add_argument('--steps', nargs='?', default = 10000,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 256,  type=int)
    parser.add_argument('--val_steps', nargs='?', default = 70,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = 'CIFAR-100', type = str)
    parser.add_argument('--checkpoint_path', type = str)
    

    args = parser.parse_args()
    if args.data_path[-1] != '/': args.data_path = args.data_path + '/'
        
    args.steps += 1
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
    


    val_metrics = {}
    val_metrics['total'] = 0
    val_metrics['CE Loss'] = 0
    val_metrics['Accuracy'] = 0
    val_metrics['Accuracy Top 5'] = 0
    
    params = checkpoints.restore_checkpoint(ckpt_dir=args.checkpoint_path, target=None)
    
    #log if applicable
    if args.log:
        wandb = wandb.init(config = args, name = args.name, project = 'CIFAR')
    else: wandb = None
    steps = 0

    train_dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)
    import time
    now = time.time()
        
    evaluator = evaluate.Evaluator(
                             params = params,
                             val_dataloader = val_dataloader,
                             args = args, 
                             validation_step = master_val_step,
                             current_step = steps,
                             val_metrics = val_metrics,
                             wandb = wandb)
    
    evaluator.evaluate()
    
    
    
    
    
    