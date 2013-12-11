from RL_framework import *
import heapq
import random

class PrioritizedSweeping(RLAlgorithm):
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

    # compute the value function V(s)
    def get_v(self, state):
        # print type(state)
        return self.V.get(state, 0)

    # get the number of times for a pair (state, action)
    def get_num_acions(self, state, action):
        return self.num_actions.get((state, action), 0)

    # compute the best reward for from a state to another
    def get_best_reward(self, state, next_state):
        actions = self.model.get_actions(state)
        reward = self.get_reward(state, actions[0], next_state)
        for action in actions:
            reward = max(reward, self.get_reward(state, action, next_state))
        return reward

    # get the next best state
    # if the best 
    def get_next_best_state(self, state):
        L = self.model.get_next_states(state)
        best_state = [L[0]]
        m = self.get_v(L[0])
        for s in L[1:]:
            # first, check for ties, then check for greater state
            if abs(self.get_v(s) - m + self.get_best_reward(state, s)) < self.delta*m:
                best_state.append(s)
            elif self.get_v(s) > m:
                m = self.get_v(s)
                best_state = [s]
        return random.choice(best_state)

    # for any state, get the best action with the highest expected reward
    # if there are actions with equal rewards, return one randomly
    def get_best_action(self, state, next_state):
        actions = self.model.get_actions(state)
        p = self.get_transition(state, actions[0], next_state)
        r = self.get_reward(state, actions[0], next_state)
        # expected reward
        m = p*r
        action = [actions[0]]
        for a in actions[1:]:
            # first check, for tie
            # then check for greater probability
            temp = self.get_transition(state, a, next_state)*self.get_reward(state, a, next_state)
            if abs(temp - m) < self.delta*m:
                action.append(a)
            elif temp > m:
                m = temp
                action = [a]
        return random.choice(action)

    # for any state, get the best action to get into that state from the current state
    # if there are actions with equal probability, choose a random one
    def get_best_action_probability(self, state, next_state):
        actions = self.model.get_actions(state)
        p = self.get_transition(state, actions[0], next_state)
        action = [actions[0]]
        for a in actions[1:]:
            # first check, for tie
            # then check for greater probability
            if abs(self.get_transition(state, a, next_state) - p) < self.delta:
                action.append(a)
            elif self.get_transition(state, a, next_state) > p:
                p = self.get_transition(state, a, next_state)
                action = [a]
        return random.choice(action)

    # update the min queue with the value & state
    def update_queue(self, state, value):
        i = -1
        # if the state is in the queue, update it
        for v in range(len(self.queue)):
            (C, s) = self.queue[v]
            if s == state:
                self.queue[v] = (min(C, value), state)
                i = v
        # if the state is not in the queue, add the state to the queue
        if i == -1:
            self.queue.append((value, state))
        #print self.queue
        heapq.heapify(self.queue)

    # compute impact C(s, s*) = sum over a P(s|s*,a)*delta(s)
    def compute_impact(self, s1, s0, delta):
        s = 0
        for action in self.model.get_actions(s0):
            s += self.get_transition(s0, action, s1)*delta
        return s

    # V(s, a) = sum over s' P(s'|s,a)*(R(s,a,s') + V(s'))
    def compute_v_per_action(self, state, action):
        s = 0
        for next_state in self.model.get_next_states(state):
            s += self.get_transition(state, action, next_state)*(
                self.get_reward(state, action, next_state) + self.get_v(next_state)*self.discount_rate)
        return s

    # perform a Bellman backup on that state
    def sweep(self, state):
        actions = self.model.get_actions(state)
        V_new = self.compute_v_per_action(state, actions[0])
        for action in actions:
            V_new = max(V_new, self.compute_v_per_action(state, action))
        delta_change = abs(self.get_v(state) - V_new)
        # update the dictionary
        self.V[state] = V_new
        # now compute the priority queue for the predecessor
        for s0 in self.model.get_prev_states(state):
            capacity = self.compute_impact(state, s0, delta_change)
            self.update_queue(s0, -capacity)

    def choose_action(self, state):
        # with some probability, choose a random action
        action = None
        if random.random() < self.epsilon:
            actions = self.model.get_actions(state)
            action = random.choice(actions)
            self.epsilon *= self.degrading_constant
            # make sure that we do still explore at the minimum level
            self.epsilon = min(self.epsilon, 0.1)
        else:
            best_next_state = self.get_next_best_state(state)
            action = self.get_best_action(state, best_next_state)
            # action = self.get_best_action_probability(state, best_next_state)
        return action

    def next(self, action = None):
        if action == None:
            action = self.choose_action(self.model.current_state)
        current_state = self.model.current_state
        reward = self.model.perform(action)
        next_state = self.model.current_state
        self.update_transition(current_state, action, next_state)
        self.update_reward(current_state, action, next_state, reward)
        self.sweep(current_state)
        for i in range(self.k - 1):
            (v, state) = heapq.heappop(self.queue)
            self.sweep(state)
        return (action, reward, next_state)