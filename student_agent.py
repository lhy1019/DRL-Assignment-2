# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import tracemalloc


from nTupleAgent import NTupleAgent, Pattern
from custom2048 import Game2048, Board
from parse_weight import load_weights
from MCTS2048 import TD_MCTS_Node, TD_MCTS

patterns = [
    Pattern([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
    Pattern([(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)]),
    Pattern([(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]),
    Pattern([(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)]),
    Pattern([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)]),
    Pattern([(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)]),
    Pattern([(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)]),
    Pattern([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]),
]
weight_file = "weights.bin"  # or sys.argv[1] if you prefer dynamic
tracemalloc.start()
weights = load_weights("weights.bin")

action_map = [0, 3, 1, 2]

agent = NTupleAgent(patterns)
agent.load_weights(weights)
mcts = TD_MCTS(agent, iterations=10, exploration_constant=1.41, rollout_depth=3, gamma=0.99)
def get_action(state, score):
    board = Board(state, score)
    if score > 10000:
        env = Game2048()
        env.reset(board)
        allowed_actions = env.getAllowedActions()
        if not allowed_actions:
            return None  # No legal move

        root = TD_MCTS_Node(copy.deepcopy(board), "action", allowed_actions)
        for _ in range(mcts.iterations):
            mcts.run_simulation(root)

        best_act, distribution = mcts.best_action_distribution(root)
    else:
        best_act = agent.best_action(board)
    action = action_map[best_act]
    return action


