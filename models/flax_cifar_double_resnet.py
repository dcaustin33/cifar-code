import flax.linen as nn
import jax.numpy as jnp
from flax_cifar_resnet import resnet18

class double_resnet(nn.Module):
    num_classes: int = 100

    def setup(self):
        self.network = resnet18()
        self.fc1 = nn.Dense(100)
        self.projection = nn.Dense(1024)
        self.embedding = nn.Embed(1, 1024)

    def __call__(self, x):
        shape = x.shape
        embed = jnp.zeros((x.shape[0],), jnp.int32)
        embed = self.embedding(embed)
        x = jnp.concatenate((x, embed), dim = 1)
        out = self.network(x)
        out1 = self.fc1(out)
        new_embedding = self.projection(out1)

        #should do this in an ipynb and make sure there are enough to copy to every single x
        new_embedding = jax.lax.reshape(new_embedding, ())

