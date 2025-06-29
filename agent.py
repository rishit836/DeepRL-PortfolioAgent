import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque,namedtuple


Stock = namedtuple("Stock",('state', 'action', 'next_state', 'reward'))
class replayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
        

    def push(self,*args):
        """save a stock"""
        self.memory.append(Stock(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class investment_agent(nn.Module):
    def __init__(self,n_days,n_actions):
        super(investment_agent, self).__init__()
