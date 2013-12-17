from RL_framework import *
from ChainModel import *
import numpy

# for a model, this class holds all the data
class Hypothesis(object):
    def __init__(self, model):
        # transition model
        self.P = {}
        # probability model
        self.R = {}
        # game model
        self.model = model
    
    def get_transition(self, state, action, next_state):
        return self.P.get((state, action), {}).get(next_state, 0)
    
    def get_reward(self, state, action):
        (u, std) = self.R.get((state, action))
        return random.gauss(u, std)
    
    def get_reward_table(self, state, action):
        return self.R[(state, action)]
    
    def get_transition_table(self, state, action, next_states):
        L = []
        for next_state in next_states:
            if self.get_transition(state, action, next_state) > 0:
                L.append((next_state, self.get_transition(state, action, next_state)))
        return L
    
    @staticmethod
    def draw_init_hypothesis(model, u0 = 0, std0 = 1):
        """ Sample an initial hypothesis for this model with u0, std0
        @model: RL_framework.Model
        """
        hypothesis = Hypothesis(model)
        # iterate through every state, action pairs
        for state in model.states:
            for action in model.actions:
                # reward model
                hypothesis.R[(state, action)] = (u0, std0)
                # transition model
                next_states = model.get_next_states(state)
                alphas = [1]*len(next_states)
                p = numpy.random.dirichlet(alphas)
                hypothesis.P[(state, action)] = {}
                for index, next_state in enumerate(next_states):
                    hypothesis.P[(state, action)][next_state] = p[index]
        return hypothesis

    @staticmethod
    def draw_hypothesis(model, keepr):
        """ Sample a hypothesis for this model with data from the keepr
        @model: RL_framework.Model
        """
        hypothesis = Hypothesis(model)
        for state in model.states:
            for action in model.actions:
                # transition model
                next_states = model.get_next_states(state)
                alphas = [1 + keepr.get_visit_count(state, action, next_state) for next_state in next_states]
                p = numpy.random.dirichlet(alphas)
                hypothesis.P[(state, action)] = {}
                for index, next_state in enumerate(next_states):
                    hypothesis.P[(state, action)][next_state] = p[index]
                # reward model
                # simplification: using sample variance instead of doing the priors
                std = (keepr.get_var_reward(state, action))**0.5
                n = keepr.get_visit_count(state, action)
                if n == 0:
                    tmp = 0.01
                    sample_mean = 0
                else:
                    tmp = max(keepr.get_var_reward(state, action)/float(n), 0.01)
                    sample_mean = float(keepr.get_sum_reward(state, action))/n
                u = numpy.random.normal(sample_mean, tmp)
                hypothesis.R[(state, action)] = (u, std)
        return hypothesis
        
#model = ChainModel()
#hypothesis = Hypothesis.get_init_hypothesis(model)
#s1 = model.state[1]
#s2 = model.state[2]
#act_a = model.act_a
#act_b = model.act_b
#print hypothesis.get_reward_table(s1, act_a)
#print hypothesis.get_reward_table(s1, act_b)
#print hypothesis.get_reward_table(s2, act_a)
#print hypothesis.get_reward_table(s2, act_b)
#print hypothesis.get_transition_table(s1, act_a, model.get_next_states(s1))


