import torch
import torch.nn as nn
import torch.optim as optim
from agent import LSTMtradingAgent,replayMemory
from data import data_gen
from env import StockMarketEnv
import random
import torch.nn.functional as F
import os
from colorama import Fore, Back, Style
from collections import deque



def train(ticker,num_episodes:int=600,save_:bool=False,verbose:bool=False,explore:bool=True):
    max_profit = 0
    current_profit = 0

    ticker = ticker
    df = data_gen(ticker)
    df.dropna(inplace=True)
    env = StockMarketEnv(df,verbose=verbose)
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
    
    print(Back.GREEN,"Starting Training",Style.RESET_ALL)
    actionmap = {0: "buy",1:"sell",2:"hold"}

    for episode in range(num_episodes):
        if explore:
            if episode < round(num_episodes*32,0):
            
                if random.randint(0,3) == 1:
                    prob_ = 2e-1
                else:
                    prob_ = 1e-2

            else:
                prob_ = 0
        else:
            prob_ = 0

        state = env.reset()
        state = env._get_state_sequence()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # converting np array to 
        total_reward = 0

        done = False
        print("-"*5,end="")
        print(f"Episode {episode+1}/{num_episodes}","-"*5)
        if verbose:
            print(Back.YELLOW,"prob set for this is:",prob_, Style.RESET_ALL)
        epoch = 0
        while not done:

            epoch += 1
            if random.random() < prob_: 
                # decaying the prob function
                prob_ -= 5e-2
                
                action = random.randint(0,1)
                if action == 1 and env.shares_held == 0:
                    action = 0
                # 0: buy, 1: sell, 2:hold
                if verbose:
                    print(Fore.WHITE, Back.YELLOW,"Exploring: Choosing a random Move:", actionmap[action],Style.RESET_ALL)
            else:
                with torch.no_grad():
                    q_values = agent(state)
                    action = q_values.argmax().item()


            next_state,reward,done = env.step_sequence(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

            memory.push(state,torch.tensor([[action]],dtype=torch.long, device=device),next_state_tensor,torch.tensor([reward], dtype=torch.float32, device=device))

            state = next_state_tensor
            total_reward += reward
            
            #check if the memory as enough data to be used as a sample
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
                print(Back.WHITE,Fore.BLACK,"Day",epoch,Style.RESET_ALL)
                if (env.cash-env.initial_cash) >=0:
                    print(Fore.GREEN,f"Agent Owns {env.cash}")
                else:
                    print(Fore.RED,f"Agent Owns {env.cash}")

            '''
            if verbose and epoch%10 != 0:
                print(Back.WHITE,Fore.BLACK,"Day",epoch,Style.RESET_ALL)
                if env.cash >0:
                    print(Fore.GREEN,f"Agent Owns {env.cash}")
                else:
                    print(Fore.RED,f"Agent Owns {env.cash}")

            '''
        if save_ and (env.cash - env.initial_cash)>0:
            if (env.cash-env.initial_cash)>max_profit:
                max_profit = env.cash-env.initial_cash

                print(Fore.GREEN, Back.WHITE,"Saving At Episode",episode,Style.RESET_ALL)
                if not os.path.exists("models/"):
                    os.mkdir("models/")
                torch.save(agent.state_dict(),"models/lstm_model_"+str(ticker)+".pth")
            else:
                print(Fore.GREEN, Back.WHITE,"Although Model is profitable but not More than models before",episode,Style.RESET_ALL)


        

        

        

    if not os.path.exists("models/"):
        os.mkdir("models/")
    torch.save(agent.state_dict(),"models/exp_lstm_model_"+str(ticker)+".pth")
    print("\n"*5)
    print(Fore.BLACK, Back.BLACK, "-"*10,Style.RESET_ALL)
    print(Fore.GREEN, Back.White,"Model Trained And Saved Succesfully",Style.RESET_ALL)
    print(Fore.BLACK, Back.BLACK, "-"*10,Style.RESET_ALL)



if __name__ == "__main__":
    train("NVDA")



