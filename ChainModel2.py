from RL_framework import *
from LoopModel import *
import random

class ChainModel2(Model):
    def __init__(self):
        self.name = "Chain Model"
        self.act_a = Action(0)
        self.act_b = Action(1)
        self.state = {}
        self.state[1] = StateNode(1, [self.act_a,self.act_b], [1,2], [2,0], [1,2,3,4,5])
        self.state[2] = StateNode(2, [self.act_a,self.act_b], [1,3], [2,0], [1])
        self.state[3] = StateNode(3, [self.act_a,self.act_b], [1,4], [2,0], [2])
        self.state[4] = StateNode(4, [self.act_a,self.act_b], [1,5], [2,0], [3])
        self.state[5] = StateNode(5, [self.act_a,self.act_b], [1,5], [2,10], [4,5])
        self.states = self.get_states_by_id([1,2,3,4,5])
        self.actions = [self.act_a, self.act_b]
        self.start_state = self.state[1]
        self.current_state = self.start_state
        self.step = 0

    def get_action_by_id(self, id):
        if id == 0:
            return self.act_a
        elif id == 1:
            return self.act_b
        else:
            return None
            
    def perform_action_a(self):
        return self.perform(self.act_a)

    def perform_action_b(self):
        return self.perform(self.act_b)

    def set_current_state_by_state_id(self, id):
        self.current_state = self.state[id]

    # perform an action on the model
    def perform(self, action):
        reward = None
        if action in self.actions:
            (next_state_id, reward) = self.current_state.perform(action)
            self.current_state = self.state[next_state_id]
            self.step += 1
        return reward

    def get_next_states(self, state):
        L = []
        for id in self.state[state.get_id()].get_next_state_ids():
            L.append(self.state[id])
        return L

    def get_states_by_id(self, L):
        states = []
        for id in L:
            states.append(self.state[id])
        return states

    def get_prev_states(self, state):
        L = []
        for id in self.state[state.get_id()].get_prev_state_ids():
            L.append(self.state[id])
        return L

    def get_actions(self, state):
        return self.actions

# define the slippery chain model where with probability e, performing an action
# will have the opposite effect
class SlipperyChainModel2(ChainModel2):
    def __init__(self, e = 0.2):
        self.e = e
        ChainModel.__init__(self)

    def perform(self, action):
        if random.random() < self.e:
            return ChainModel.perform(self, self.get_action_by_id(1 - action.id))
        else:
            return ChainModel.perform(self, action)


# m = SlipperyChainModel(0.1)
# a = m.act_a
# b = m.act_b
# print m.perform(a)
# print m.current_state
# print m.perform(a)
