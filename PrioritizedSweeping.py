from RL_framework import *

class PrioritizedSweeping(RLAlgorithm):
	# model: the input model
	# e: 
	def __init__(self, model, e = 0):
		self.model = model
		# reward model
		self.R = {}
		# transition model
		self.P = {}
		# value model
		self.V = {}

	def get_reward(self, s1, a, s2):
		v = (s1, a, s2)
		if v in self.R:
			(s, total) = self.R[v]
			return s/float(total)
		return 0

	def get_transition(self, s1, a, s2):
		v = (s1, a, s2)
		if v in self.P:
			return self.P[v]/float(self.P[(s1, a)])
		return 0

	def get_v(self, state):
		return self.V.get(state, 0)

	def get_next_best_state(self):
		L = self.model.get_next_states(self.current_state)
		best_state = L[0]
		m = self.get_v(best_state)

		for s in sef.model.get_next_states(self.current_state):
			if self.get_v(s) > m:
				m = self.get_v(s)
				best_state = s
		return best_state

	def get_best_action(self, next_state):
		actions = self.model.get_actions()

		action = actions[0]
		p = self.get_transition(self.model.current_state, action, next_state)

		for a in actions:
			if self.get_transition(self.model.current_state, a, next_state) > p:
				p = self.get_transition(self.model.current_state, a, next_state)
				action = a
		return action

	def update_transition(self, s1, a, s2):
		self.P[(s1, a)] = self.P.get((s1, a), 0) + 1
		self.P[(s1, a, s2)] = self.P.get((s1, a, s2), 0) + 1

	def update_reward(self, s1, a, s2, r):
		(s, total) = self.R.get((s1, a, s2), (0, 0))
		self.R[(s1, a, s2)] = (s + r, total + 1)

	def sweep(self, state):
		actions = self.model.get_actions(state)
		def calculate()

	def next(self):
		best_next_state = self.get_next_best_state()
		action = self.get_best_action(best_next_state)
		current_state = self.model.current_state
		reward = self.model.perform(action)
		next_state = self.model.current_state
		self.update_transition(s1, a, s2)
		self.update_reward(s1, a, s2)