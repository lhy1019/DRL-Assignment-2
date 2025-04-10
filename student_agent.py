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
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)

action_map = [0, 3, 1, 2]

agent = NTupleAgent(patterns)
agent.load_weights(weights)
def get_action(state, score):
    board = Board(state, score)
    action = agent.best_action(board)
    if action == -1:
        return random.randint(0, 3)
    action = action_map[action]
    return action


