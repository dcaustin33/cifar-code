#gneeric python imports
import argparse
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb

#custom imports
from utils import  top_1_error_rate_metric, top_5_error_rate_metric
import haiku as hk
import cifar_100
import haiku_trainer as trainer
from haiku_cifar_resnet import resnet18
#from haiku_resnets import ResNet18 as resnet18
import cifar_100

#python helper inputs
import time
from typing import NamedTuple


def prepare_data(dataset_args, val_dataset_args):
    
    train_dataset = cifar_100.CIFAR_100_transformations(train = True, **dataset_args)
    dataloader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=256,
        num_workers=8,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True
    )

    val_dataset = cifar_100.CIFAR_100_transformations(views = 1, train = True, **val_dataset_args)
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

class TrainState(NamedTuple):
  params: hk.Params
  state: hk.State
  optimizer: optax._src.base.GradientTransformation
  opt_state: optax.OptState


def _forward(image, is_training: bool) -> jnp.ndarray:
  """Forward application of the resnet."""
  net = resnet18()
  return net(image, is_training=is_training)


def create_params(example, rng):
    def resnet(x):
        resnet = resnet18()
        return resnet(x, True)
    resnet = hk.transform_with_state(resnet)
    params = resnet.init(rng, example)
    return params


def create_optimizer(optimizer, lr, wd):
    optimizer = optimizer(lr)
    return optimizer

def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

def master_train_step(state, data, metrics):
    params = state.params
    #print(params)
    #print(state.state)

    logits, loss, grads, new_state = training_step(state.state, params, data)
    updates, state.opt_state = state.optimizer.update(grads, state.opt_state, state.params)
    state.params = optax.apply_updates(state.params, updates)

    acc1 = top_1_error_rate_metric(logits = logits, one_hot_labels = data['label']) 
    acc5 = top_5_error_rate_metric(logits = logits, one_hot_labels = data['label']) 

    metrics['total'] += data['image0'].shape[0]
    metrics['Loss'] += loss
    metrics['Accuracy'] += acc1
    metrics['Accuracy Top 5'] += acc5

    state.state = new_state

    return state

def loss_fn(state, params, data):
    logits, state = forward.apply(params, state, None, data['image0'], is_training=True)
    loss = jnp.mean(jax.vmap(cross_entropy_loss)(logits=logits, labels=data['label']), axis = 0)
    return loss, (loss, logits, state)

@jax.jit
def training_step(state, params, data):
    print('in')
    time.sleep(1)
    grads, (loss, logits, new_state)= (jax.grad(loss_fn, has_aux=True)(state, params, data))
    logits, state = forward.apply(params, state, None, data['image0'], is_training=True)
    loss, _ = loss_fn(state, params, data)
    print(logits, logits.dtype)
    print(loss, loss.dtype)
    print('out')
    time.sleep(1)
    return logits, loss, grads, new_state


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
    schedule = optax.warmup_cosine_decay_schedule(
                                                    init_value=3e-05,
                                                    peak_value=args.lr,
                                                    warmup_steps=args.warmup_steps,
                                                    decay_steps=args.steps,
                                                    end_value=args.lr * .01,
                                                    )

    #resnet = hk.transform(resnet18)
    #params = resnet.init(rng, jnp.ones((4, 32, 32, 3)))
    
    forward = hk.transform_with_state(_forward)
    params, state = forward.init(init_rng, jnp.ones((4, 32, 32, 3)), True)
    optimizer = optax.adamw(schedule, weight_decay=args.wd)
    opt_state = optimizer.init(params)

    state = TrainState(params, state, optimizer, opt_state)
    # Transform our forwards function into a pair of pure functions.
    


    #prepare the data
    #dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)

    #log if applicable
    if args.log and args.rank == 0:
        wandb = wandb.init(config = args, name = args.name, project = 'CIFAR')
    else: wandb = None
    steps = 0

    train_dataset, dataloader, val_dataloader = prepare_data(args.dataset_args, args.val_dataset_args)
    import time
    now = time.time()
     
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
    
    
    
    
    
    