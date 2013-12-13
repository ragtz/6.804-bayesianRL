# this class keeps auxiliary information for RL algorithm
# TODO: write test suits
class Keeper(object):
    def __init__(self):
        self.sum_reward_squares = {}
        self.sum_reward = {}
        self.visit_count_state = {}
        self.visit_count_state_action = {}
        # reward model
        self.R = {}
        # transition model
        self.P = {}

    def get_sum_reward(self, state, action):
        return self.sum_reward.get((state, action), 0)

    def update_sum_reward(self, state, action, r):
        self.sum_reward[(state, action)] = self.get_sum_reward(state, action) + r

    def get_sum_reward_squares(self, state, action):
        return self.sum_reward_squares.get((state, action), 0)

    def update_sum_reward_squares(self, state, action, r):
        self.sum_reward_squares[(state, action)] = self.get_sum_reward_squares(state, action) + r**2

    def get_visit_count(self, state, action = None):
        if action == None:
            return self.visit_count_state.get(state, 0)
        return self.visit_count_state_action.get((state, action), 0)

    def increase_count(self, state, action):
        self.visit_count_state[state] = self.get_visit_count(state) + 1
        self.visit_count_state_action[(state, action)] = self.get_visit_count(state, action) + 1

    def get_reward(self, s1, a, s2):
        """ compute the reward for state, action, next state"""
        v = (s1, a, s2)
        if v in self.R:
            (s, total) = self.R[v]
            return float(s)/total
        return 0

    def update_reward(self, s1, a, s2, r):
        """ Update the reward model"""
        (s, total) = self.R.get((s1, a, s2), (0, 0))
        self.R[(s1, a, s2)] = (s + r, total + 1)

    def get_reward_table(self, state, action, next_states):
        L = []
        for next_state in next_states:
            if self.get_reward(state, action, next_state) > 0:
                L.append((next_state, self.get_reward(state, action, next_state)))
        return L

    # compute transition function P(s1, a, s2)
    def get_transition(self, s1, a, s2):
        v = (s1, a, s2)
        if v in self.P:
            return self.P[v]/float(self.get_visit_count(s1, a))
        return 0

    # update the transition model, keeping track of counts
    def update_transition(self, s1, a, s2):
        self.increase_count(s1, a)
        self.P[(s1, a, s2)] = self.P.get((s1, a, s2), 0) + 1

    def get_transition_table(self, state, action, next_states):
        L = []
        for next_state in next_states:
            if self.get_transition(state, action, next_state) > 0:
                L.append((next_state, self.get_transition(state, action, next_state)))
        return L