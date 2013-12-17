from PrioritizedSweeping import *
import numpy
from RL_framework import Model
from Hypothesis import *
from PriorityQueue import UniquePriorityQueue
    
class BayesPrioritizedSweeping(RLAlgorithm):
    def __init__(self, model, k = 2 , discount_rate = 0.9):
        self.model = model
        self.discount_rate = discount_rate
        self.keepr = Keeper()
        # priority queue for ML
        self.ML_queue = UniquePriorityQueue()
        # comparison constant
        self.delta = 0.001
        # number of back-up per action
        self.k = k
        # default mean and std
        self.u0 = 0
        self.std0 = 1
        # draw initial hypothesis
        self.hypothesis = Hypothesis.draw_init_hypothesis(model, self.u0, self.std0)
        # maximum-likelihood V
        self.ML_V = {}
            
    #def get_transition(self, s1, a, s2):
        #return self.hypothesis.get_transition(s1, a, s2)
    
    def get_ML_transition(self, s1, a, s2):
        return RLAlgorithm.get_transition(self, s1, a, s2)
    
    #def get_reward(self, s1, a, s2):
        #return self.hypothesis.get_reward(s1, a)
    
    def get_ML_reward(self, s1, a, s2):
        return RLAlgorithm.get_reward(self, s1, a, s2)
    
    def get_ML_v(self, state):
        return self.ML_V.get(state, 0)
    
    # compute impact C(s, s*) = sum over a P(s|s*,a)*delta(s)
    # s1: current state, s0: predecessor
    def compute_impact(self, s1, s0, delta, transition_func):
        s = 0
        for action in self.model.get_actions(s0):
            s += transition_func(s0, action, s1)*delta
        return s    
    
    # V(s, a) = sum over s' P(s'|s,a)*(R(s,a,s') + V(s')*discount_rate)
    def compute_v_per_action(self, state, action, transition_func, reward_func, v_func):
        s = 0
        for next_state in self.model.get_next_states(state):
            s += transition_func(state, action, next_state)*(
                reward_func(state, action, next_state) + v_func(next_state)*self.discount_rate**2)
        return s
    
    # perform a Bellman backup on that state, 
    def sweep(self, state):
        actions = self.model.get_actions(state)
        V_new = self.compute_v_per_action(state, actions[0], self.get_ML_transition,
                                          self.get_ML_reward, self.get_ML_v)
        for action in actions[1:]:
            V_new = max(V_new, self.compute_v_per_action(
                state, action, self.get_ML_transition, self.get_ML_reward, self.get_ML_v))
        delta_change = abs(self.get_ML_v(state) - V_new)
        # update the dictionary
        self.ML_V[state] = V_new
        # now compute the priority queue for the predecessor
        for s0 in self.model.get_prev_states(state):
                capacity = self.compute_impact(state, s0, delta_change, self.get_ML_transition)
                self.ML_queue.push_or_update(-capacity, s0)

    # sweep the Bellman queue    
    def sweep_queue(self):
        for i in range(self.k - 1):
            (priority, state) = self.ML_queue.pop()
            self.sweep(state)            
    
    def draw_hypothesis(self):
        self.hypothesis = Hypothesis.draw_hypothesis(self.model, self.keepr)        
    
    def choose_action(self, state):
        return random.choice(self.model.get_actions(state))
        
    def next(self, action=None):
        if self.model.current_state == self.model.start_state:
            # draw a new hypothesis
            self.draw_hypothesis()     
        current_state = self.model.current_state
        if action == None:
            action = self.choose_action(current_state)
        reward = self.model.perform(action)
        next_state = self.model.current_state
        # do book-keeping
        self.keepr.update_reward_and_transition(current_state, action, next_state, reward)
        self.sweep(current_state)
        self.sweep_queue()        
        return (action, reward, next_state)
        
    def get_transition(self, s1, a, s2):
        raise Exception("this method is discontinued")
    
    def get_reward(self, s1, a, s2):
        raise Exception("this method is discontinued")