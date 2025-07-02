import numpy as np
import pandas as pd
from colorama import just_fix_windows_console,init
from colorama import Fore, Back, Style
just_fix_windows_console()
init()



class StockMarketEnv:
    def __init__(self,df:pd.DataFrame, initial_cash:float=10000,window_size:int = 14,verbose:bool=False):
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.window_size = window_size
        self.investment = 0
        self.reset()
        self.verbose = verbose
        

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares_held = 0
        self.investment = 0
        self.prev_cash = self.initial_cash
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
            # row['MACD_Bullish_Crossover'],
            # row['MACD_Bearish_Crossover'],
            self.cash,
            self.shares_held,
            self.investment,
            self.total_value]
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

        self.prev_cash = self.cash
        self.prev_total_value = self.total_value
        self.prev_investment= self.investment

        if action == 0  and self.cash>=price:
            self.shares_held+=1
            self.cash-= price
            self.investment+= price

        elif action == 1 and self.shares_held>0:
            self.shares_held -= 1
            self.cash += price
            self.investment-= price


        elif action == 1 and self.shares_held<=0:
            self.reward  = -2
            done = self.current_step>= len(self.df) - 1

            return self._get_state_sequence(),self.reward, done

            
        self.current_step +=1



        done = self.current_step>= len(self.df) - 1

        self.total_value = self.cash + self.investment
        self.investment_change = self.prev_investment-self.investment
        
        
        if ((self.cash- self.prev_cash)/self.cash) >0:
            self.reward = 5
            self.prev_cash = self.cash # setting a new milestone for the agent to pass 

            if self.verbose:
                print("-"*10)
                print(Fore.LIGHTGREEN_EX,"agent made a profit of $",(self.cash- self.initial_cash),Style.RESET_ALL)
                print("-"*10)
        elif ((self.cash - self.prev_cash)/self.cash) <0  and action != 0:
            self.prev_cash  = self.cash
            self.reward = -1
        else:
            self.reward = 0

        # so the agent doesnt learn to just hold onto the cash it has and not take any action
        if not done and self.current_step > 90 and self.cash == self.initial_cash:
            self.reward = -5
            
        if self.verbose:
            print("-"*5)
            print(Fore.BLUE,f"Step: {self.current_step}, Action: {action}, Reward: {self.reward}, Total Value: {self.total_value}",Style.RESET_ALL)
            print("-"*5)
        if self.verbose:
            if done :
                print("-"*5)
                if self.cash>0:
                    print(Fore.GREEN,f"agent owns right now {self.cash}",Style.RESET_ALL)
                else:
                    print(Fore.RED,f"agent owns right now {self.cash}",Style.RESET_ALL)
                if (self.cash -self.initial_cash)>0:
                    print(Fore.GREEN,"Agent made $", self.cash -self.initial_cash,Style.RESET_ALL)
                else:
                    print(Fore.RED,"Agent made $", self.cash -self.initial_cash,Style.RESET_ALL)

                print("-"*5)




        return self._get_state_sequence(),self.reward, done





