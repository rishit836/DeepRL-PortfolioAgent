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



class LSTMTradingAgent(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMTradingAgent, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]              
        out = self.fc(out)              
        return out
