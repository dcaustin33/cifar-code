import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
import time
import os
from flax.training import train_state, checkpoints
from flax_logger import log_metrics as logger

class Evaluator:
    def __init__(self, 
                 params: train_state,
                 val_dataloader: DataLoader,
                 args,
                 validation_step,
                 current_step = None,
                 val_metrics: dict = None,
                 wandb = None):
        
        self.params = params
        self.val_dataloader = val_dataloader
        self.args = args
        self.validation_step = validation_step
        self.current_step = current_step
        self.val_metrics = val_metrics
        self.wandb = wandb

    def convert_data(self, data):
        data['label'] = jnp.array(data['label'])
        data['label'] = jax.nn.one_hot(data['label'], num_classes=100)
        data['image0'] = jnp.array(data['image0'].permute(0, 2, 3, 1))

        return data
            
    def evaluate(self):

        now = time.time()

        if self.current_step: steps = self.current_step
        else: steps = 0


        print('Starting from step', steps)
        print('Validation for', self.args.val_steps, 'steps')
            
        check_path = 'checkpoints/' + self.args.name
        check_path = check_path + '/'
        val_steps = 0
        while val_steps < self.args.val_steps:

            for k, val_data in enumerate(self.val_dataloader):

                val_data = self.convert_data(val_data)
                if val_steps >= self.args.val_steps: break

                _ = self.validation_step(self.params, val_data, self.val_metrics)
                if val_steps % self.args.log_n_val_steps == 0 and val_steps != 0:
                    self.val_metrics = logger(self.val_metrics, steps, wandb = self.wandb, train = False)
                val_steps += 1

        print(time.time() - now)