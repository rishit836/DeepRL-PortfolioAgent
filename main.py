from train import train
import sys

if __name__ == "__main__":
    try: 
        ticker = sys.argv[1]
    except:
        ticker = "NVDA"
    train(ticker)
