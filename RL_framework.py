class RLObject:
	def __init__(self, name):
		self.name = name

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)  

class Action(RLObject):
	def __str__(self):
		return "Action " + self.name

class State(RLObject):
	def __str__(self):
		return "State " + self.name

class Model(RLObject):
	def __init__(self, name):
		self.name = name
		# list of all actions
		self.actions = []
		# list of all states
		self.states = []
		self.current_state = None

	def __str__(self):
		return "Model:" + self.name 

	def get_current_state(self):
		raise Exception("not implemented")

	# get all available actions for this state
	def get_actions(self):
		raise Exception("not implemented")

	# get the available next states from this state - might
	def get_next_states(self):
		raise Exception("not implemented")

	def perform_action(self, action):
		raise Exception("not implemented")

a = Action("1")
b = Action("1")
c = Action("2")
d = State("!")
e = State("!")

print a
print a == b
print a == c
print a == d
print e == d