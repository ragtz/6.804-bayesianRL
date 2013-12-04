from RL_framework import *
from ChainModel import *
import random

class QLearning(RLAlgorithm):
    # model: the input model
    # e: the parameter for randomization
    def __init__(self, model, learning_rate = 0.2, discount_rate = 0.2, e = 0.2):
        self.model = model
        # reward model
        self.R = {}
        # transition model
        self.P = {}
        # value model
        self.V = {}
        # parameters for the algorithm
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.e = e
        self.Q = {}
        # the difference constant - used to check if two quantities are roughly the same
        self.detla = 0.001

    # return the quality function
    def get_Q(self, state, action):
        return self.Q.get((state, action), 0)

    def get_max_Q(self, state):
        m = 0
        for action in self.model.get_actions(state):
            m = max(m, self.get_Q(state, action))
        return m

    # for any state, get the best action by computing the quality function  Q(state, action)
    def get_best_action(self, state):
        actions = self.model.get_actions(self.model.current_state)
        m = self.get_Q(state, actions[0])
        best_action = [actions[0]]
        for action in actions:
            # first, check for ties
            if abs(self.get_Q(state, action) - m) < self.detla:
                best_action.append(action)
            elif self.get_Q(state, action) > m:
                m = self.get_Q(state, action)
                best_action = [action]
        return random.choice(best_action)

    # update the quality function
    def update_Q(self, s1, a, s2, r):
        q = self.get_Q(s1, a)
        if (s1, a) in self.Q:
            self.Q[(s1, a)] = q + self.learning_rate * (r + self.discount_rate * self.get_max_Q(s2) - q)
        else:
            self.Q[(s1, a)] = r

    def next(self, action = None):
        if action == None:
            # with some probability, choose a random action
            if random.random() < self.e:
                actions = self.model.get_actions(self.model.current_state)
                action = random.choice(actions)
            else:
                action = self.get_best_action(self.model.current_state)
        current_state = self.model.current_state
        reward = self.model.perform(action)
        next_state = self.model.current_state
        self.update_Q(current_state, action, next_state, reward)
        return (action, reward, next_state)
        
