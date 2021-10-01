import os
import sys
from PyQt5 import QtWidgets

from alphazero.manager import Manager 
from alphazero.network import Network
from alphazero.misc import PrintColors, load_model

from args import args

if __name__ == "__main__":
    game_set = ("Connect 4", "TicTacToe", "Gomoku", "Othello")
    print(f"Selected game: {PrintColors.bold}{args.game}{PrintColors.endc}\n")
    if args.game == "Connect 4":
        from alphazero.games.connect4.connect4_rules import Connect4Rules as Rules
        from alphazero.games.connect4.connect4_network import Connect4Network as Network
        from alphazero.games.connect4.connect4_gui import Connect4Gui as Gui
    elif args.game == "TicTacToe":
        from alphazero.games.tictactoe.tictactoe_rules import TicTacToeRules as Rules
        from alphazero.games.tictactoe.tictactoe_network import TicTacToeNetwork as Network
        from alphazero.games.tictactoe.tictactoe_gui import TicTacToeGui as Gui
    elif args.game == "Gomoku":
        from alphazero.games.gomoku.gomoku_rules import GomokuRules as Rules
        from alphazero.games.gomoku.gomoku_network import GomokuNetwork as Network
        from alphazero.games.gomoku.gomoku_gui import GomokuGui as Gui
    elif args.game == "Othello":
        from alphazero.games.othello.othello_rules import OthelloRules as Rules
        from alphazero.games.othello.othello_network import OthelloNetwork as Network
        from alphazero.games.othello.othello_gui import OthelloGui as Gui
    else:
        raise NotImplementedError(f"Game '{args.game}' not implemented. Implemented games are: {game_set}")

    # Create the game rules object
    print("Creating game rules object")
    if args.game == "Gomoku":
        rules = Rules(args.gomoku_size)
    else:
        rules = Rules()

    # Create network
    print("Creating neural network\n")
    network = Network(args)
    
    # If we are playing against a model
    if args.play:
        print(f"Loading model: '{PrintColors.bold}{args.play}{PrintColors.endc}'")
        load_model(network, "models", args.play)
        print(f"{PrintColors.green}Model loaded successfully{PrintColors.endc}")

        print("\nStarting game GUI\n")
        app = QtWidgets.QApplication(sys.argv)
        gui = Gui(rules, network, args)
        sys.exit(app.exec_())

    # If we are training a model
    else:
        # If we are starting with a given model
        if args.model:
            print(f"Loading model: '{PrintColors.bold}{args.model}{PrintColors.endc}'")
            if not os.path.isfile(os.path.join("models", args.model)):
                print(f"{PrintColors.red}Error: cannot find model 'models/{args.model}'.{PrintColors.endc}\nStarting training with a newly initialized model.")
            else:
                load_model(network, "models", args.model)
            print(f"{PrintColors.green}Model loaded successfully{PrintColors.endc}\n")

        print("Creating manager object and starting training\n")
        manager = Manager(rules, network, args)
        manager.train()
