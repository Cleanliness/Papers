import jax
import flax
from flax import linen as nn

class ResBlock(nn.Module):
    features: int
    conv: nn.Module = nn.Conv
    act: nn.Module = nn.relu
    project: nn.Module = nn.Dense
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x,):
        residual = x
        x = self.conv(self.features, (3, 3), (1, 1), padding='SAME', use_bias=self.use_bias)(x)
        x = self.act(x)
        x = self.conv(self.features, (3, 3), (1, 1), padding='SAME', use_bias=self.use_bias)(x)
        x = x + self.project(self.features)(residual) 
        x = self.act(x)
        return x

# resnet = ResBlock(features=64)
# print(resnet.tabulate(jax.random.PRNGKey(0), jax.numpy.ones((1, 32, 32, 3))))

