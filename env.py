import numpy as np
import pandas as pd

class StockMarketEnv:
    def __init__(self,df:pd.DataFrame, initial_cash:float=10000,window_size:int = 14):
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.window_size = window_size
        self.reset()
        

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares_held = 0
        self.total_value = self.cash
        return self._get_state()
    
    '''
    this works for normal layer model but in lstm model we need to have
    sequence of states instead so new function is needed
    '''
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
    

    def _get_state_sequence(self,):
        # create a padding incase we dont have enough histroy
        start = self.current_step - self.window_size
        end = self.current_step


        sequence = []
        history = self.df.iloc[start:end]
        for _,row in history.iterrows():
            features = [row['Close'],
            row['RSI_14'],
            row['MACD'],
            row['MACD_Signal'],
            row['MACD_Hist'],
            row['MACD_Bullish_Crossover'],
            row['MACD_Bearish_Crossover'],
            self.cash,
            self.shares_held]
            sequence.append(features)
        return np.array(sequence,dtype=np.float32)
        
         
    
    def step(self,action):
        price = self.df.loc[self.current_step,'Close']
        prev_total_value = self.cash + self.shares_held * price

        # 0: buy, 1: sell, 2:hold
        if action == 0  and self.cash>=price:
            self.shares_held+=1
            self.cash-=price

        elif action == 1 and self.shares_held>0:
            self.shares_held -= 1
            self.cash += price
        self.current_step +=1

        done = self.current_step>= len(self.df) - 1

        self.total_value = self.cash + self.shares_held*price
        reward = self.total_value - prev_total_value # so the model instead of maximazing net worth and portfolio learns to maximise short term gains

        return self._get_state(),reward, done


    def step_sequence(self,action):
        price = self.df.loc[self.current_step,'Close']
        prev_total_value = self.cash + self.shares_held * price
        if action == 0  and self.cash>=price:
            self.shares_held+=1
            self.cash-= price

        elif action == 1 and self.shares_held>0:
            self.shares_held -= 1
            self.cash += price
        self.current_step +=1

        done = self.current_step>= len(self.df) - 1

        self.total_value = self.cash + self.shares_held*price
        reward = self.total_value - prev_total_value # so the model instead of maximazing net worth and portfolio learns to maximise short term gains
        reward -= 0.01 # transaction fees type shit

        return self._get_state_sequence(),reward, done





