import jax
import jax.numpy as jnp
import flax.linen as nn
from torch.utils.data import DataLoader
import time
import os
from flax.training import train_state, checkpoints
import optax
from utils.flax_logger import log_metrics as logger

class Trainer:
    def __init__(self, 
                 state: train_state,
                 dataloader: DataLoader, 
                 val_dataloader: DataLoader,
                 args, 
                 training_step,
                 validation_step,
                 current_step = None,
                 metrics: dict = None,
                 val_metrics: dict = None,
                 wandb = None):
        
        self.state = state
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.args = args
        self.training_step = training_step
        self.validation_step = validation_step
        self.current_step = current_step
        self.metrics = metrics
        self.val_metrics = val_metrics
        self.wandb = wandb

    def convert_data(self, data):
        data['label'] = jnp.array(data['label'])
        data['label'] = jax.nn.one_hot(data['label'], num_classes=100)
        data['image0'] = jnp.array(data['image0'].permute(0, 2, 3, 1))

        return data
            
    def train(self):

        now = time.time()

        if self.current_step: steps = self.current_step
        else: steps = 0

        print('Starting from step', steps)
        print('Training for', self.args.steps, 'steps')
            
        check_path = 'checkpoints/' + self.args.name
        check_path = check_path + '/'

        while steps < self.args.steps:

            for i, data in enumerate(self.dataloader):
                if steps >= self.args.steps: break
                steps += 1
                #print(steps)
                data = self.convert_data(data)

                self.state = self.training_step(self.state, data, self.metrics)

                if steps % self.args.log_n_train_steps == 0:
                    self.metrics = logger(self.metrics, steps, wandb = self.wandb, train = True)
                if steps % 10 == 0:
                    print(steps, time.time() - now) 
                
                if steps % self.args.log_n_steps == 0:
                    checkpoints.save_checkpoint(ckpt_dir='{name}'.format(name = check_path), 
                                                target=self.state, step=steps, overwrite=True,
                                                prefix = 'Checkpoint_{name}'.format(name = self.args.name))

                    val_steps = 0
                    while val_steps < self.args.val_steps:

                        for k, val_data in enumerate(self.val_dataloader):

                            val_data = self.convert_data(val_data)
                            if val_steps >= self.args.val_steps: break

                            _ = self.validation_step(self.state, val_data, self.val_metrics)
                            if val_steps % self.args.log_n_val_steps == 0 and val_steps != 0:
                                self.val_metrics = logger(self.val_metrics, steps, wandb = self.wandb, train = False)
                            val_steps += 1

        checkpoints.save_checkpoint(ckpt_dir='{name}'.format(name = check_path), 
                                    target=self.state.params, step=steps, overwrite=True,
                                    prefix = 'Final_{name}'.format(name = self.args.name))

        print(time.time() - now)