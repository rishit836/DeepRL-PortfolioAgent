import yfinance as yf
import os
import sys
import numpy as np
import pandas as pd


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

def data_gen(ticker):
    if not os.path.exists(ticker+"_data.csv"):
        t = yf.Ticker(ticker)
        data = t.history(period="1y")
        df = pd.DataFrame(data)
        df['date'] = df.index
        df.reset_index(inplace=True,drop=True)
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.to_csv(ticker+"_data.csv",index=False)
    else:
        df = pd.read_csv(ticker+"_data.csv")
        df['SMA_14'] = calculate_SMA(df,14)
        df['EMA_14'] = calculate_ema(df,14)
        df['RSI_14'] = calculate_rsi(df,14)
        print(df.head(30))
        





if __name__ == "__main__":
    try:
        ticker = sys.argv[1]
    except:
        ticker = "NVDA"
    data_gen(ticker)