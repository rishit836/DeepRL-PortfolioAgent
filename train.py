import torch
import torch.nn as nn
import torch.optim as optim
from agent import LSTMtradingAgent,replayMemory
from data import data_gen
from env import StockMarketEnv
import random
import torch.nn.functional as F
import os

def train(ticker,num_episodes:int=600,save_:bool=False):
    ticker = ticker
    df = data_gen(ticker,True)
    df.dropna(inplace=True)
    env = StockMarketEnv(df)
    n_features = env._get_state_sequence().shape[1] # getting number of features dynamically
    n_actions = 3 # buy/sell/hold
    memory = replayMemory(10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = LSTMtradingAgent(n_actions=n_actions ,n_features=n_features,n_layers=12).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    batch_size = 64
    gamma = .99
    epsilon = 1
    epsilon_min = .01
    epsilon_decay = .999
    prob_ = 7e-1
    print("Starting Training")
    for episode in range(num_episodes):
        state = env.reset()
        state = env._get_state_sequence()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # converting np array to 
        total_reward = 0

        done = False
        print("-"*5,end="")
        print(f"Episode {episode+1}/{num_episodes}",end="")
        print("-"*5)
        epoch = 0
        while not done:

            epoch += 1
            if random.random() < prob_:
                action = random.randint(0,2)
                # decay
                prob_ -= 1e-3
            else:
                with torch.no_grad():
                    q_values = agent(state)
                    action = q_values.argmax().item()


            next_state,reward,done = env.step_sequence(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            memory.push(state,torch.tensor([[action]],dtype=torch.long, device=device),next_state_tensor,torch.tensor([reward], dtype=torch.float32, device=device))

            state = next_state_tensor
            total_reward += reward

            if len(memory) >batch_size:
                batch = memory.sample(batch_size)
                batch_state, batch_action,batch_next_state, batch_reward  = zip(*batch)

                batch_state = torch.cat(batch_state)
                batch_action = torch.cat(batch_action)
                batch_next_state = torch.cat(batch_next_state)
                batch_reward = torch.cat(batch_reward)

                q_values = agent(batch_state).gather(1, batch_action)

                with torch.no_grad():
                    next_q_values = agent(batch_next_state).max(1)[0]
                    expected_q = batch_reward + gamma * next_q_values

                loss = F.mse_loss(q_values.squeeze(), expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch%10 == 0:
                print("Day",epoch)
                print("Agent Owns",end=" ")
                print(env.total_value)

        if save_:
            print("Saving At Episode",episode)
            if not os.path.exists("models/"):
                os.mkdir("models/")
            torch.save(agent.state_dict(),"models/lstm_model_"+str(ticker)+".pth")

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        

        

    if not os.path.exists("models/"):
        os.mkdir("models/")
    torch.save(agent.state_dict(),"models/lstm_model_"+str(ticker)+".pth")
    print("\n"*5)
    print("-"*10)
    print("Model Trained And Saved Succesfully")
    print("-"*10)


if __name__ == "__main__":
    train("NVDA")



