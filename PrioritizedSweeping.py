from RL_framework import *

class PrioritizedSweeping(RLAlgorithm):
	def __init__(self, model):
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
			return self.R[v]
		return 0

	def get_transition(self, s1, a, s2):
		v = (s1, a, s2)
		if v in self.P:
			return self.P[v]
		return 0

	def get_v(self, state):
		if state in self.V:
			return self.V[state]
		return 0

	def get_next_best_state(self):
		L = self.model.get_next_states()
		best_state = L[0]
		m = self.get_v(best_state)

		for s in sef.model.get_next_states():
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

	def next(self):
		best_next_state = self.get_next_best_state()
		action = self.get_best_action(best_next_state)
		current_state = self.model.current_state
		reward = self.model.perform(action)
		next_state = self.model.current_state

		return []

