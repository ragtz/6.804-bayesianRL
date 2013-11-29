from RL_framework import *

class ChainModel(Model):
	def __init__(self):
		self.name = "Chain Model"
		self.state = {}
		self.state[1] = State(1)
		self.state[2] = State(2)
		self.state[3] = State(3)
		self.state[4] = State(4)
		self.state[5] = State(5)
		self.act_a = Action(0)
		self.act_b = Action(1)

		self.states = [self.state[1], self.state[2], self.state[3],
			self.state[4], self.state[5]]

		self.actions = [self.act_a, self.act_b]

		self.current_state = self.state[1]
		self.step = 0

	def perform(self, action):
		next_state = None
		reward = None

		if action == self.act_a:
			if self.current_state == self.state[5]:
				next_state = self.state[5]
				reward = 10
			else:
				id = self.current_state.get_id()
				next_state = self.state[id + 1]
				reward = 0
		else:
			next_state = self.state[1]
			reward = 2

		self.current_state = next_state
		self.step += 1

		return reward

m = ChainModel()
a = m.act_a
b = m.act_b
print m.perform(a)
print m.current_state
print m.perform(a)
print m.current_state
print m.perform(a)
print m.current_state
print m.perform(a)
print m.current_state
print m.perform(a)
print m.current_state
