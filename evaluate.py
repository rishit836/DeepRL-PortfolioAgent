import torch
from agent import LSTMtradingAgent
from data import data_gen
from env import StockMarketEnv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def evaluate_model(ticker):
    path = "models/lstm_model_"+str(ticker)+".pth"

    df=data_gen(ticker)
    df.dropna(inplace=True)
    env = StockMarketEnv(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = env._get_state_sequence().shape[1]
    n_actions = 3
    model = LSTMtradingAgent(n_actions=n_actions, n_features=n_features, n_layers=12)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)

    state = env.reset()
    state = env._get_state_sequence()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    done = False
    actions_taken = []
    portfolio_values = []

    while not done:
        with torch.no_grad():
            q_values = model(state)
            action = q_values.argmax().item()
        
        actions_taken.append(["BUY", "SELL", "HOLD"][action])
        portfolio_values.append(env.total_value)

        next_state, reward, done = env.step_sequence(action)
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

    print("\nEvaluation Complete!")
    print("Final Portfolio Value: $", round(env.total_value, 2))
    print("Initial Cash: $", env.initial_cash)
    print("Net Profit/Loss: $", round(env.total_value - env.initial_cash, 2))
    print("\nActions Taken:")
    print(actions_taken)

    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        ticker = sys.argv[1]
    except:
        ticker = "NVDA"
    if os.path.exists("models/lstm_model_"+str(ticker)+".pth"):
        evaluate_model(ticker)
    else:
        print("Model Needs to be trained first!")



        
        
