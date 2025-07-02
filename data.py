import yfinance as yf
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_SMA(data: pd.DataFrame, window :int =14)-> pd.Series:
    return data['Close'].rolling(window).mean()

def calculate_ema(data:pd.DataFrame, window:int=14) -> pd.Series:
    return data['Close'].ewm(span=window, adjust=False).mean()
def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder’s smoothing
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_MACD(data:pd.DataFrame):
    ema_12 = calculate_ema(data,window=12)
    ema_26 = calculate_ema(data,window=26)
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9,adjust=False).mean()
    histogram = macd_line - signal_line
    

    return macd_line,signal_line,histogram



def data_gen(ticker,verbose:bool=False):
    if not os.path.exists("data/"+str(ticker)+"_data.csv"):
        t = yf.Ticker(ticker)
        data = t.history(period="1y")
        df = pd.DataFrame(data)
        df['date'] = df.index
        df.reset_index(inplace=True,drop=True)
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        if not os.path.exists("data"):
            os.mkdir("data")
        df.to_csv("data/"+str(ticker)+"_data.csv",index=False)
        
        if verbose:
            print("data downloaded")
    else:
        df = pd.read_csv("data/"+str(ticker)+"_data.csv")
        if verbose:
            print("data exists.")

    if verbose:
        print("generating Indicators..")
    df['SMA_14'] = calculate_SMA(df,14)
    df['EMA_14'] = calculate_ema(df,14)
    df['RSI_14'] = calculate_rsi(df,14)
    df['MACD'],df['MACD_Signal'],df['MACD_Hist']=calculate_MACD(df)

    '''
    to let model learn the pattern below :-
    MACD Crosses Above Signal	Bullish Signal (Buy)
    MACD Crosses Below Signal	Bearish Signal (Sell)

    when MACD > 0 → price bullish (short-term > long-term)
    when MACD < 0 → price bearish
    '''



    df['MACD_Bullish_Crossover'] = ((df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
    df['MACD_Bearish_Crossover'] = ((df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))).astype(int)
    
    if verbose:
        print("data is done")
    return df[['Open', 'High', 'Low', 'Close', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'MACD_Signal','MACD_Hist', 'MACD_Bullish_Crossover', 'MACD_Bearish_Crossover']]

    
        





if __name__ == "__main__":
    try:
        ticker = sys.argv[1]
    except:
        ticker = "NVDA"
    df = data_gen(ticker)
    plt.boxplot(df['Close'])
    plt.show()
    print(df.columns)