from train import train
import sys

if __name__ == "__main__":
    try: 
        ticker = sys.argv[1]
        save_best = sys.argv[2]
        verbose = sys.argv[3]
        explore = sys.argv[4]
    except:
        ticker = "NVDA"
        save_best  = True
        verbose = True
        explore = True
    train(ticker,num_episodes=100,save_=save_best,verbose=verbose,explore=explore)
