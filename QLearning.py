from RL_framework import *
import heapq
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

	# return the quality function
	def get_Q(self, state, action):
		return self.Q.get((state, action), 0)

	def get_max_Q(self, state):
		m = 0
		for action in self.model.get_actions(state):
			m = max(m, self.get_Q(state, action))
		return m

	# compute reward function R(s1, a, s2)
	def get_reward(self, s1, a, s2):
		v = (s1, a, s2)
		if v in self.R:
			(s, total) = self.R[v]
			return s/float(total)
		return 0

	def get_transition_table(self, state, action):
		L = []
		for next_state in self.model.get_next_states(state):
			if self.get_transition(state, action, next_state) > 0:
				L.append((next_state, self.get_transition(state, action, next_state)))
		return L

	def get_reward_table(self, state, action):
		L = []
		for next_state in self.model.get_next_states(state):
			if self.get_reward(state, action, next_state) > 0:
				L.append((next_state, self.get_reward(state, action, next_state)))
		return L

	# compute transition function P(s1, a, s2)
	def get_transition(self, s1, a, s2):
		v = (s1, a, s2)
		if v in self.P:
			return self.P[v]/float(self.P[(s1, a)])
		return 0

	# for any state, get the best action by computing the quality function  Q(state, action)
	def get_best_action(self, state):
		actions = self.model.get_actions(self.model.current_state)
		best_action = actions[0]
		m = self.get_Q(state, best_action)
		for action in actions:
			if self.get_Q(state, action) > m:
				m = self.get_Q(state, action)
				best_action = action
		return best_action

	# update the transition model, keeping track of counts
	def update_transition(self, s1, a, s2):
		self.P[(s1, a)] = self.P.get((s1, a), 0) + 1
		self.P[(s1, a, s2)] = self.P.get((s1, a, s2), 0) + 1

	# keeping track of the reward model
	def update_reward(self, s1, a, s2, r):
		(s, total) = self.R.get((s1, a, s2), (0, 0))
		self.R[(s1, a, s2)] = (s + r, total + 1)

	# update the quality function
	def update_Q(self, s1, a, s2, r):
		q = self.get_Q(s1, a)
		self.Q[(s1, a)] = q + self.learning_rate * (r + self.discount_rate * self.get_max_Q(s2) - q)

	def next(self, action = None):
		if action == None:
			# with some probability, choose a random action
			if random.random() < self.e:
				actions = self.model.get_actions(self.model.current_state)
				action = actions[random.randint(0, len(actions) - 1)]
			else:
				action = self.get_best_action(self.model.current_state)
		current_state = self.model.current_state
		reward = self.model.perform(action)
		next_state = self.model.current_state
		self.update_transition(current_state, action, next_state)
		self.update_reward(current_state, action, next_state, reward)
		self.update_Q(current_state, action, next_state, reward)
		return (reward, next_state)

ps = QLearning(SlipperyChainModel())
for i in range(1000):
	print ps.next()
# expect state 5 to have the highest potential
for state in ps.model.states:
    for action in ps.model.actions:
	    print "("+str(state)+","+str(action)+"): "+str(ps.get_Q(state, action))

# for i in range(1, 6):
# 	print "transition model"
# 	print ps.get_transition_table(ps.model.state[i], ps.model.act_a)
# 	print ps.get_transition_table(ps.model.state[i], ps.model.act_b)
# 	print "reward model"
# 	print ps.get_reward_table(ps.model.state[i], ps.model.act_a)
# 	print ps.get_reward_table(ps.model.state[i], ps.model.act_b)
