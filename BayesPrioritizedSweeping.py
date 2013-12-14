from PrioritizedSweeping import *
from numpy import random

# for a model, this class holds all the data
class Hypothesis(object):
    def __init__(self):
        # transition model
        self.P = {}
        # probability model
        self.R = {}
    
    def get_transition(self, state, action, next_state):
        return self.P.get((state, action), {}).get(next_state, 0)
    
    def get_reward(self, state, action):
        (u, std) = self.R.get((state, action))
        return random.gauss(u, std)

class BayesPrioritizedSweeping(PrioritizedSweeping):
    def __init__(self, model, discount_rate = 0.9):
        self.model = model
        self.discount_rate = discount_rate
        self.hypothesis = Hypothesis()
        self.keepr = Keeper()
    
    def get_transition(self, s1, a, s2):
        return self.hypothesis.get_transition(s1, a, s2)
    
    def get_ML_transition(self, s1, a, s2):
        return PrioritizedSweeping.get_transition(self, s1, a, s2)
    
    def get_reward(self, s1, a, s2):
        return self.hypothesis.get_reward(s1, a)
    
    def get_ML_reward(self, s1, a, s2):
        return PrioritizedSweeping.get_reward(self, s1, a, s2)
    
    def next(self, action=None):
        if self.model.current_state == self.model.start_state:
            # draw a new hypothesis
            pass
        
        current_state = self.model.current_state
        action = self.choose_action(current_state)
        reward = self.model.perform(action)
        next_state = self.model.current_state
        
        return (action, reward, next_state)
        