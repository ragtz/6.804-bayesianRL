from RL_framework import *

class StateNode(State):
    def __init__(self, id, actions, next_state_ids, rewards, prev_state_ids):
        self.id = id
        self.next_state_ids = next_state_ids
        self.prev_state_ids = prev_state_ids
        self.asr = {}
        for i, a in enumerate(actions):
            self.asr[str(a)] = (next_state_ids[i],rewards[i])
            
    def perform(self, action): 
        return self.asr[str(action)]
        
    def get_next_state_ids(self):
        return self.next_state_ids
        
    def get_prev_state_ids(self):
        return self.prev_state_ids


class LoopModel(Model):
    def __init__(self):
        self.name = "Loop Model"
        self.act_a = Action(0)
        self.act_b = Action(1)
        self.state = {}
        self.state[0] = StateNode(0, [self.act_a,self.act_b], [1,5], [0,0], [4,5,6,7,8])
        self.state[1] = StateNode(1, [self.act_a,self.act_b], [2,2], [0,0], [0])
        self.state[2] = StateNode(2, [self.act_a,self.act_b], [3,3], [0,0], [1])
        self.state[3] = StateNode(3, [self.act_a,self.act_b], [4,4], [0,0], [2])
        self.state[4] = StateNode(4, [self.act_a,self.act_b], [0,0], [1,1], [3])
        self.state[5] = StateNode(5, [self.act_a,self.act_b], [0,6], [0,0], [0])
        self.state[6] = StateNode(6, [self.act_a,self.act_b], [0,7], [0,0], [5])
        self.state[7] = StateNode(7, [self.act_a,self.act_b], [0,8], [0,0], [6])
        self.state[8] = StateNode(8, [self.act_a,self.act_b], [0,0], [2,2], [7])
        self.states = self.get_states_by_id([0,1,2,3,4,5,6,7,8])
        self.actions = [self.act_a, self.act_b]
        self.current_state = self.state[0]
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
            self.current_state = self.states[next_state_id]
            self.step += 1
        return reward

    def get_next_states(self, state):
        L = []
        for id in self.states[state.get_id()].get_next_state_ids():
            L.append(self.states[id])
        return L

    def get_states_by_id(self, L):
        states = []
        for id in L:
            states.append(self.state[id])
        return states

    def get_prev_states(self, state):
        L = []
        for id in self.states[state.get_id()].get_prev_state_ids():
            L.append(self.states[id])
        return L

    def get_actions(self, state):
        return self.actions
        
