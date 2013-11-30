from RL_framework import *
import heapq
from ChainModel import *
import random

class PrioritizedSweeping(RLAlgorithm):
	# model: the input model
	# e: the parameter for randomization
	def __init__(self, model, k = 5, e = 0.5):
		self.model = model
		# reward model
		self.R = {}
		# transition model
		self.P = {}
		# value model
		self.V = {}
		# parameters for the algorithm
		self.k = k
		self.e = e
		# priority queue
		self.queue = []

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

	# compute the value function V(s)
	def get_v(self, state):
		# print type(state)
		return self.V.get(state, 0)

	# get the next best state
	def get_next_best_state(self):
		L = self.model.get_next_states(self.model.current_state)
		best_state = L[0]
		m = self.get_v(best_state)
		for s in self.model.get_next_states(self.model.current_state):
			if self.get_v(s) > m:
				m = self.get_v(s)
				best_state = s
		return best_state

	# for any state, get the best action to get into that state from the current state
	def get_best_action(self, next_state):
		actions = self.model.get_actions(self.model.current_state)
		action = actions[0]
		p = self.get_transition(self.model.current_state, action, next_state)
		for a in actions:
			if self.get_transition(self.model.current_state, a, next_state) > p:
				p = self.get_transition(self.model.current_state, a, next_state)
				action = a
		return action

	# update the transition model, keeping track of counts
	def update_transition(self, s1, a, s2):
		self.P[(s1, a)] = self.P.get((s1, a), 0) + 1
		self.P[(s1, a, s2)] = self.P.get((s1, a, s2), 0) + 1

	# keeping track of the reward model
	def update_reward(self, s1, a, s2, r):
		(s, total) = self.R.get((s1, a, s2), (0, 0))
		self.R[(s1, a, s2)] = (s + r, total + 1)

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
				self.get_reward(state, action, next_state) + self.get_v(next_state))
		return s

	# perform a Bellman backup on that state
	def sweep(self, state):
		actions = self.model.get_actions(state)
		V_new = self.compute_v_per_action(state, actions[0])
		for action in actions:
			V_new = max(V_new, self.compute_v_per_action(state, action))
		delta = abs(self.get_v(state) - V_new)
		# update the dictionary
		self.V[state] = V_new
		for s0 in self.model.get_prev_states(state):
			capacity = self.compute_impact(state, s0, delta)
			self.update_queue(s0, -capacity)

	def next(self, action = None):
		if action == None:
			# with some probability, choose a random action
			if random.random() < self.e:
				actions = self.model.get_actions(self.model.current_state)
				action = actions[random.randint(0, len(actions) - 1)]
			else:
				best_next_state = self.get_next_best_state()
				action = self.get_best_action(best_next_state)
		current_state = self.model.current_state
		reward = self.model.perform(action)
		next_state = self.model.current_state
		self.update_transition(current_state, action, next_state)
		self.update_reward(current_state, action, next_state, reward)
		self.sweep(current_state)
		for i in range(self.k - 1):
			(v, state) = heapq.heappop(self.queue)
			self.sweep(state)
		return (reward, next_state)

ps = PrioritizedSweeping(SlipperyChainModel(), 3, 0.1)
for i in range(10000):
	print ps.next()
# expect state 5 to have the highest potential
for state in ps.model.states:
	print ps.get_v(state)

# for i in range(1, 6):
# 	print "transition model"
# 	print ps.get_transition_table(ps.model.state[i], ps.model.act_a)
# 	print ps.get_transition_table(ps.model.state[i], ps.model.act_b)
# 	print "reward model"
# 	print ps.get_reward_table(ps.model.state[i], ps.model.act_a)
# 	print ps.get_reward_table(ps.model.state[i], ps.model.act_b)
