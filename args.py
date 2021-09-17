from alphazero.misc import Arguments

args = Arguments({
    # How many training iterations to perform.
    "iterations"            : 500,           

    # How many self-play games to play per iteration.
    "episodes"              : 150,          

    # How many training examples to store in memory at a time.
    "replay_memory"         : 200_000,      

    # How many evaluation matches between the new and previous neural network checkpoints.
    "eval_matches"          : 50,           

    # What percentage of the eval games must the updated nnet win to be accepted.
    "acceptance_threshold"  : 0.55,         

    "temperature"           : 1.0,         
    "cpuct"                 : 1.0,          

    # Number of leaf-position rollouts for each move.
    "monte_carlo_sims"      : 160,           
    
    # Number of residual blocks in the network.
    # This must match the number of residual blocks in any network being loaded
    # from the models/ directory as they are being loaded.
    "residual_blocks"       : 14,            
    
    # Network learning rate.
    "learning_rate"         : 0.001,        

    # Number of epochs to train the neural network per iteration.
    "epochs"                : 12,           

    # Batch size used when updating the neural network.
    "batch_size"            : 128,          

    # Enable cuda acceleration for the neural network.
    "cuda"                  : True,         

    # The name of a model you want to play against. The model must be in the
    # models/ directory.
    # The 'residual_blocks' setting must match the number of residual blocks in
    # the network used for the match.
    "play"                  : None,         
    
    # The name of a model you want the training to start from.
    "model"                 : "othello-14block",         

    # 'Connect 4', 'TicTacToe', 'Gomoku', 'Othello'
    "game"                  : "Othello",   

    # Size of the gomoku board.
    "gomoku_size"           : 13,                
})
