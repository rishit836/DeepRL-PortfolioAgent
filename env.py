import numpy as np
import pandas as pd

class StockMarketEnv:
    def __init__(self,df:pd.DataFrame, initial_cash:float=10000):
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.total_value = self.cash
        return self._get_state()
    def _get_state(self):
        # Getting Various indicators for the current day/current step (2025-06-17)
        state = [
            self.df.loc[self.current_step,'Close'],
            self.df.loc[self.current_step,'RSI_14'],
            self.df.loc[self.current_step,'MACD'],
            self.df.loc[self.current_step,'MACD_Signal'],
            self.df.loc[self.current_step,'MACD_Hist'],
            self.df.loc[self.current_step,'MACD_Bullish_Crossover'],
            self.df.loc[self.current_step,'MACD_Bearish_Crossover'],
            self.cash,
            self.shares_held]
        
        return np.array(state,dtype=np.float32)
    
    def step(self,action):
        price = self.df.loc[self.current_step,'Close']

        # 0: buy, 1: sell, 2:hold
        if action == 0  and self.cash>=price:
            self.shares_held+=1
            self.cash-=self.price

        elif action == 1 and self.shares_held>0:
            self.shares_held -= 1
            self.cash += self.price
        self.current_step +=1

        done = self.current_step>= len(self.df) - 1

        self.total_value = self.cash + self.shares_held*self.price
        reward = self.total_value - self.initial_cash

        return self._get_state(),reward, done




