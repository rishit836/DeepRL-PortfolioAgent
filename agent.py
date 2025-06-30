import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque,namedtuple


Stock = namedtuple("Stock",('sequence', 'action', 'next_sequence', 'reward'))
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



class LSTMtradingAgent(nn.Module):
    def __init__(self,n_actions,n_features,n_layers,hidden_size:int=64):
        super(LSTMtradingAgent,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_layers

        self.lstm = nn.LSTM(n_features,self.hidden_size,num_layers=n_layers,batch_first=True)
        self.fc = nn.Linear(self.hidden_size, n_actions)

    def forward(self,x):
        # x.size(0) = batch_size that is how many timesteps are given at once (how many days)

        #  these layers are initially set to 0 for LSTM hidden states and cell states
        #  .to incase GPU accelration available pytorch takes care of that and thus h0 and c0 are on same device
        h0= torch.zeros(self.num_layers, x.size(0),self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)

        out = out[:, -1, :]  # Take the last timestep = (batch_size, hidden_size)

        out = self.fc(out)   # Final prediction = (batch_size, n_actions)

        return out


    # the bot can act and explore new actions i.e not only hold to buying but also sell and hold and learn accordingly
    def act(self,state,epsilon,device):
        if random.random()>epsilon:
            # we dont want any gradient we just want it learn from inferences
            with torch.no_grad():
                state = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to_device(device)
                q_values = self.forward(state)
                return torch.argmax(q_values).item() # picks the action with highest Q value (most rewarding)
        else:
            return random.randrange(self.fc.out_features) #choose any random action from actions

        
