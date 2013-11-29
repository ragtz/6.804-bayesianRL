class RLObject:
	def __init__(self, id):
		self.id = id

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)  

	def get_id(self):
		return self.id

class Action(RLObject):
	def __str__(self):
		return "Action " + str(self.id)

class State(RLObject):
	def __str__(self):
		return "State " + str(self.id)

class Model(RLObject):
	def __init__(self, name):
		self.name = name
		# list of all actions
		self.actions = []
		# list of all states
		self.states = []
		self.current_state = None
		self.step = 0

	def __str__(self):
		s = "Model:" + self.name + "\n"
		s += "States:" + str(self.states) + "\n"
		s += "Actions:" + str(self.actions) + "\n"

		return s 

	def get_current_state(self):
		raise Exception("not implemented")

	# get all available actions for this state
	def get_actions(self):
		raise Exception("not implemented")

	# get the available next states from this state - might
	def get_next_states(self):
		raise Exception("not implemented")

	def perform(self, action):
		raise Exception("not implemented")

	def num_steps(self):
		return self.step