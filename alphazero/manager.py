import time
from collections import deque

from alphazero.selfplay import selfplay
from alphazero.evaluate import evaluate
from alphazero.rules import Rules
from alphazero.network import Network
from alphazero.misc import PrintColors, Arguments, session_setup, save_checkpoint, load_checkpoint, get_time_stamp

class Manager():
    """
    AlphaZero manager class running the training pipeline.
    """
    def __init__(self, rules: Rules, network: Network, args: Arguments):
        self.rules = rules
        self.args = args

        self.network = network
        self.checkpoint_network = self.network.__class__(self.args)
        
        self.session_number = session_setup(self.rules, self.args)
        self.checkpoint_number = 0
        save_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)
        load_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)

        self.training_examples = deque(maxlen=self.args.replay_memory)

    def train(self) -> None:
        """
        Iterate the training pipeline for [arguments.iterations] iterations.
        """
        for i in range(self.args.iterations):
            print(f"{PrintColors.transparent}-----{PrintColors.endc} {PrintColors.bold}Iteration: {i + 1}/{self.args.iterations}{PrintColors.endc} {PrintColors.transparent}-----{PrintColors.endc}\n")
            self.iterate()
    
    def iterate(self) -> None:
        """
        Performs a single iteration of the training pipeline.
        """
        st = time.time()

        print(f"{PrintColors.yellow}Self-Play Data Generation{PrintColors.endc}")
        training_examples = selfplay(self.rules, self.network, self.args)
        self.training_examples.extend(training_examples)

        # Train the network
        print(f"\n{PrintColors.yellow}Training Neural Network{PrintColors.endc}")
        self.network.train(self.training_examples)

        # Pit the new and previous network against each other to evaluate updated performance.
        # Only if the updated network has improved significantly will it be saved as the next checkpoint
        print(f"\n{PrintColors.yellow}Evaluation{PrintColors.endc}")
        wins, ties, losses = evaluate(self.rules, self.network, self.checkpoint_network, self.args)
        
        score = wins / max((wins + losses), 1)
        print(f"W: {PrintColors.green}{wins}{PrintColors.endc} - T: {ties} - L: {PrintColors.red}{losses}{PrintColors.endc} - Score: {PrintColors.bold}{round(score, 3)}{PrintColors.endc}")

        if score > self.args.acceptance_threshold:
            print(f"Checkpoint {PrintColors.green}Accepted{PrintColors.endc}")
            self.checkpoint_number += 1
            save_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)
            load_checkpoint(self.checkpoint_network, self.session_number, self.checkpoint_number, self.args)
        else:
            print(f"Checkpoint {PrintColors.red}Discarded{PrintColors.endc}")
            load_checkpoint(self.network, self.session_number, self.checkpoint_number, self.args)

        t = get_time_stamp(time.time() - st)
        print(f"\nIteration time: {t}")
        print(f"Checkpoint: {PrintColors.bold}{self.checkpoint_number}{PrintColors.endc}\n\n")

