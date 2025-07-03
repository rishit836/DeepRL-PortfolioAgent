from train import train
import sys

if __name__ == "__main__":
    try: 
        ticker = sys.argv[1]
        print(ticker)
    except:
        ticker = "NVDA"
    ticker = ticker.upper()

    try:
        normalize = sys.argv[5]
    except:
        normalize = False

    try:
        save_best = int(sys.argv[2])
    except:
        save_best  = True
            
    try:
        verbose = int(sys.argv[3])
        
    except:
        verbose = True


    try:
        explore = int(sys.argv[4])
    except:
        explore = True

        
    
    print(f"Config for model training ticker:{ticker}, save_best:{bool(save_best)}, verbose:{bool(verbose)}, explore mode:{bool(explore)},normalize={bool(normalize)}")
    train(ticker,num_episodes=100,save_=bool(save_best),verbose=bool(verbose),explore=bool(explore),normalize=bool(normalize))


