class Arguments(dict):
    def __getattr__(self, attr):
        return self[attr]

args = Arguments({
    # AlphaZero
    "iterations"            : 25,           
                                            # how many training iterations to perform

    "episodes"              : 120,          
                                            # how many self-play games to play per iteration

    "play_memory"           : 200_000,      
                                            # how many game positions to store in memory at a time

    "eval_matches"          : 50,           
                                            # how many eval matches between the new and previous nnet checkpoints

    "acceptance_threshold"  : 0.55,         
                                            # what percentage of the eval games must the updated nnet win to be accepted

    "temperature"           : 1.0,         
    "cpuct"                 : 1.0,          

    "monte_carlo_sims"      : 25,           
                                            # number of node rollouts for each move

    # Nnet
    "residual_blocks"       : 2,            
                                            # number of residual blocks in the nnet.
                                            # this must match the number of residual blocks in any nnet being loaded
                                            # from /trained-models/ as they are being loaded

    "learning_rate"         : 0.001,        
                                            # nnet learning rate

    "epochs"                : 10,           
                                            # num epochs to train the nnet per iteration

    "batch_size"            : 128,          
                                            # nnet update batch size

    "cuda"                  : True,         
                                            # enable cuda

    # Models
    "duel"                  : None,         
                                            # the name of a model you want to play against. The model must be in
                                            # trained-models/
                                            # the 'residual_blocks' setting must match the number of residual blocks in
                                            # the network used for the duel.

    "model"                 : None,         
                                            # the name of a model you want the training to start from

    # Game and game-size
    "game"                  : "gomoku",   
                                            # 'connect4', 'tictactoe', 'gomoku', 'othello'

    "gomoku_size"           : 7,                
                                            # size of the gomoku board

    "othello_size"          : 8             
                                            # size of the othello board
})
