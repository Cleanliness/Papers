import jax
import flax.linen as nn
from resblock import ResBlock

class ResNet(nn.Module):
    """
    Identical to architecture defined in original ResNet paper
    """
    stack_s_size: int = 3
    stack_m_size: int = 4
    stack_l_size: int = 6
    num_classes: int = 10
    pool: nn.Module = nn.avg_pool
    linear: nn.Module = nn.Dense

    def setup(self):
        self.stack_s = nn.Sequential(
            [ResBlock(64) for _ in range(self.stack_s_size)]
        )
        self.stack_m = nn.Sequential(
            [ResBlock(128) for _ in range(self.stack_m_size)]
        )
        self.stack_l = nn.Sequential(
            [ResBlock(256) for _ in range(self.stack_l_size)]
        )

        # output logits
        self.fc_final = nn.Dense(self.num_classes) 
    

    def __call__(self, x):

        B = x.shape[0]

        x = self.stack_s(x)
        x = self.stack_m(x)

        x = self.stack_l(x)
        x = self.pool(x, (2, 2), (2, 2))
        x = x.reshape((B, -1))
        x = self.fc_final(x)
        return x
