import copy
import random
import math
import numpy as np

from nTupleAgent import NTupleAgent, Pattern
from custom2048 import Game2048, Board
from parse_weight import load_weights

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, type, allowed_actions, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = copy.deepcopy(state)
        self.type = type
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        if type == "action":
            self.untried_actions = allowed_actions
        else:
            self.untried_actions = []
        

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0

        


class TD_MCTS:
    def __init__(self, agent, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.agent = agent
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state):
        env = Game2048()
        env.board = copy.deepcopy(state)
        env.score = state.score
        return env

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        # select child with highest average values
        for action, child in node.children.items():
            ucb = (child.total_reward / child.visits) + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def rollout(self, sim_env):
        total_reward = 0.0
        board_value = self.agent.value(sim_env.board)
        sim_env.spawn_tile()
        best_action = self.agent.best_action(sim_env.board)
        afterstate = sim_env.simulateAfterstate(best_action)
        sim_env.board = copy.deepcopy(afterstate)
        v_afterstate = self.agent.value(sim_env.board)
        r = sim_env.step(best_action)
        total_reward += r + 0.5*v_afterstate + board_value
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state)

        # Selection
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            afterstate = sim_env.simulateAfterstate(node.action)
            sim_env.board = copy.deepcopy(afterstate)
            sim_env.score = afterstate.score

            if sim_env.check_game_over():
                return

        # Expansion
        if not node.fully_expanded():
            action = node.untried_actions.pop()
            afterstate = sim_env.simulateAfterstate(action)
            sim_env.board = copy.deepcopy(afterstate)
            sim_env.score = afterstate.score

            allowed_actions = sim_env.getAllowedActions()
            child_node = TD_MCTS_Node(
                state=copy.deepcopy(sim_env.board),
                allowed_actions=allowed_actions,
                type = "afterstate",
                parent=node,
                action=action
            )
            node.children[action] = child_node
            node = child_node

        # Rollout
        reward = self.rollout(sim_env)
        self.backpropagate(node, reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
