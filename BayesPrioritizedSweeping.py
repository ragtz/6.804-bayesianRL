from PrioritizedSweeping import *
import numpy
from RL_framework import Model
from Hypothesis import *
    
class BayesPrioritizedSweeping(RLAlgorithm):
    def __init__(self, model, k = 2 , discount_rate = 0.9):
        self.model = model
        self.discount_rate = discount_rate
        self.hypothesis = Hypothesis()
        self.keepr = Keeper()
        # priority queue
        self.queue = []
        # comparison constant
        self.delta = 0.001
        # number of back-up per action
        self.k = k
        # default mean and std
        self.u0 = 0
        self.std0 = 1
            
    def get_transition(self, s1, a, s2):
        return self.hypothesis.get_transition(s1, a, s2)
    
    def get_ML_transition(self, s1, a, s2):
        return PrioritizedSweeping.get_transition(self, s1, a, s2)
    
    def get_reward(self, s1, a, s2):
        return self.hypothesis.get_reward(s1, a)
    
    def get_ML_reward(self, s1, a, s2):
        return PrioritizedSweeping.get_reward(self, s1, a, s2)
    
    def get_ML_v(self, state):
        return self.V.get(state, 0)
    
    # compute impact C(s, s*) = sum over a P(s|s*,a)*delta(s)
    # s1: current state, s0: predecessor
    def compute_impact(self, s1, s0, delta):
        s = 0
        for action in self.model.get_actions(s0):
            s += self.get_ML_transition(s0, action, s1)*delta
        return s    
    
    # V(s, a) = sum over s' P(s'|s,a)*(R(s,a,s') + V(s')*discount_rate)
    def compute_v_per_action(self, state, action):
        s = 0
        for next_state in self.model.get_next_states(state):
            s += self.get_ML_transition(state, action, next_state)*(
                self.get_ML_reward(state, action, next_state) + self.get_v(next_state)*self.discount_rate**2)
        return s
    
    # perform a Bellman backup on that state
    def sweep(self, state):
        actions = self.model.get_actions(state)
        V_new = self.compute_v_per_action(state, actions[0])
        for action in actions[1:]:
            V_new = max(V_new, self.compute_v_per_action(state, action))
        delta_change = abs(self.get_v(state) - V_new)
        # update the dictionary
        self.V[state] = V_new
        # now compute the priority queue for the predecessor
        for s0 in self.model.get_prev_states(state):
                capacity = self.compute_impact(state, s0, delta_change)
                self.update_queue(s0, -capacity)

    # sweep the Bellman queue    
    def sweep_queue(self):
        for i in range(self.k - 1):
            (v, state) = heapq.heappop(self.queue)
            self.sweep(state)            
    
    # V(s, a) = sum over s' P(s'|s,a)*(R(s,a,s') + V(s')*discount_rate)    
    def compute_ML_v_per_action(self, state, action):
        s = 0
        for next_state in self.model.get_next_states(state):
            s += self.get_transition(state, action, next_state)*(
            self.get_reward(state, action, next_state) + self.get_v(next_state)*self.discount_rate**2)
        return s
    
    def draw_hypothesis():
        
    
    def next(self, action=None):
        if self.model.current_state == self.model.start_state:
            # draw a new hypothesis
            pass
        
        current_state = self.model.current_state
        action = self.choose_action(current_state)
        reward = self.model.perform(action)
        next_state = self.model.current_state
        # do book-keeping
        self.keepr.update_transition(current_state, action, next_state)
        self.keepr.update_reward(current_state, action, next_state, reward)
        self.sweep(current_state)
        self.sweep_queue()
        
        return (action, reward, next_state)
        
    def get_transition(self, s1, a, s2):
        raise Exception("this method is discontinued")
    
    def get_reward(self, s1, a, s2):
        raise Exception("this method is discontinued")