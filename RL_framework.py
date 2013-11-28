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

# class Model: 
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