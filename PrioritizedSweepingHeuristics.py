from RL_framework import *
from PrioritizedSweeping import *
import heapq
import random

class PrioritizedSweepingHeuristics(PrioritizedSweeping):
    # model: the input model
    # e: the parameter for randomization
    def __init__(self, model, k = 2, epsilon = 0.90, degrading_constant = 0.99, discount_rate = 0.9):
        self.model = model
        # reward model
        self.R = {}
        # transition model
        self.P = {}
        # value model
        self.V = {}
        # parameters for the algorithm
        self.k = k
        self.epsilon = epsilon
        self.degrading_constant = degrading_constant
        self.discount_rate = discount_rate
        # keep track of the number of (state, action)
        self.num_actions = {}
        # priority queue
        self.queue = []
        self.delta = 0.001

    # get the number of times for a pair (state, action)
    def get_num_actions(self, state, action):
        return self.num_actions.get((state, action), 0)

    # get the least performed action for this state
    def get_least_action(self, state):
        actions = self.model.actions
        m = self.get_num_actions(state, actions[0])
        least_action = [actions[0]]
        # first, check for tie
        # then find the least performed action
        for action in actions[1:]:
            if self.get_num_actions(state, action) == m:
                least_action.append(action)
            elif self.get_num_actions(state, action) < m:
                m = self.get_num_actions(state, action)
                least_action = [action]
        return (random.choice(least_action), 4.0/(4 + m**2))

    def update_action(self, state, action):
        self.num_actions[(state, action)] = self.get_num_actions(state, action) + 1

    def choose_action(self, state):
        # with some probability, choose a random action
        (least_action, epsilon) = self.get_least_action(state)
        # print epsilon
        action = None
        if random.random() < epsilon:
            self.update_action(state, least_action)
            action = least_action
        else:
            best_next_state = self.get_next_best_state(state)
            action = self.get_best_action(state, best_next_state)
            # action = self.get_best_action_probability(state, best_next_state)
        return action