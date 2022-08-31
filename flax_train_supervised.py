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
import flax_trainer as trainer
from flax.training import train_state
from flax_cifar_resnet import resnet18
import cifar_100_data


def prepare_data(dataset_args, val_dataset_args):
    
    train_dataset = cifar_100_data.CIFAR_100_transformations(train = True, **dataset_args)
    dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    val_dataset = cifar_100_data.CIFAR_100_transformations(views = 1, train = True, **val_dataset_args)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        num_workers=args.workers,
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

def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()


def create_train_state(rng, optimizer):
    """Creates initial `TrainState`."""
    model = resnet18()
    batch = jnp.ones((4, 32, 32, 3))  # (N, H, W, C) format
    params = model.init(jax.random.PRNGKey(0), batch)['params']
    tx = optimizer
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def loss_fn(params, data):
    logits, _ = model.apply({'params': params}, data['image0'], mutable=['batchstats'])
    loss = jnp.mean(jax.vmap(cross_entropy_loss)(logits=logits, labels=data['label']), axis= 0)
    return loss, logits

def master_train_step(state, data, metrics):

    state, logits, loss = training_step(state, data)

    acc1 = top_1_error_rate_metric(logits = logits, one_hot_labels = data['label']) 
    acc5 = top_5_error_rate_metric(logits = logits, one_hot_labels = data['label']) 

    metrics['total'] += data['image0'].shape[0]
    metrics['Loss'] += loss
    metrics['Accuracy'] += acc1
    metrics['Accuracy Top 5'] += acc5

    return state



@jax.jit
def training_step(state, data):
    
    def loss_fn(params, data):
        logits, _ = resnet18().apply({'params': params}, data['image0'], mutable=['batch_stats'])
        loss = jnp.mean(jax.vmap(cross_entropy_loss)(logits=logits, labels=data['label']), axis= 0)
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, data)
    state = state.apply_gradients(grads=grads)

    return state, logits, loss


def master_val_step(state, data, metrics):
    logits = validation_step(state, data)
    acc1 = top_1_error_rate_metric(logits = logits, one_hot_labels = data['label']) 
    acc5 = top_5_error_rate_metric(logits = logits, one_hot_labels = data['label']) 

    metrics['total'] += data['image0'].shape[0]
    metrics['Accuracy'] += acc1
    metrics['Accuracy Top 5'] += acc5

    return None


@jax.jit
def validation_step(state, data):
    
    logits, _ = resnet18().apply({'params': state.params}, data['image0'], mutable=['batch_stats'])
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
    parser.add_argument('--name', default = 'Flax_CIFAR_100_Supervised', type = str)
    parser.add_argument('--lr', nargs='?', default = .001, type=float)
    parser.add_argument('--workers', nargs='?', default = 1,  type=int)
    parser.add_argument('--steps', nargs='?', default = 10000,  type=int)
    parser.add_argument('--batch_size', nargs='?', default = 256,  type=int)
    parser.add_argument('--val_steps', nargs='?', default = 70,  type=int)
    parser.add_argument('--log_n_steps', nargs='?', default = 1000,  type=int)
    parser.add_argument('-log', action='store_true')
    parser.add_argument('--data_path', default = 'CIFAR-100', type = str)
    parser.add_argument('--gpus', default = 1, type = int)
    parser.add_argument('--log_n_train_steps', default = 100, type = int)
    parser.add_argument('--warmup_steps', default = 200, type = int)
    parser.add_argument('-checkpoint', action='store_true')
    parser.add_argument('--checkpoint_path', default = None, type = str)
    args = parser.parse_args()

    if args.data_path[-1] != '/': args.data_path = args.data_path + '/'
        
    args.wd = 1.5e-6
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
    schedule = optax.warmup_cosine_decay_schedule(
                                                    init_value=3e-05,
                                                    peak_value=args.lr,
                                                    warmup_steps=args.warmup_steps,
                                                    decay_steps=args.steps,
                                                    end_value=args.lr * .01,
                                                    )

    optimizer = optax.adamw(schedule, weight_decay=args.wd)
    state = create_train_state(init_rng, optimizer)

    #log if applicable
    if args.log and args.rank == 0:
        wandb = wandb.init(config = args, name = args.name, project = 'CIFAR')
    else: wandb = None
    steps = 0

    train_dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)

    trainer = trainer.Trainer(
                             state = state,
                             dataloader = dataloader, 
                             val_dataloader = val_dataloader,
                             args = args, 
                             training_step = master_train_step,
                             validation_step = master_val_step,
                             current_step = steps,
                             metrics = metrics,
                             val_metrics = val_metrics,
                             wandb = wandb)
    
    trainer.train()
    