from text import symbols

class hparams: 
    seed = 0 
    
    #data parameters 
    text_cleaners = ['english_cleaners']
    
    # model parameters
    n_symbols = len(symbols) 
    symbols_embedding_dim = 512