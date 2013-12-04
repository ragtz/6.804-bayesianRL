from LoopModel import *
from ChainModel import *
from PrioritizedSweeping import *

model = LoopModel()
ps = PrioritizedSweeping(model, k = 10, e = 0.9)

total = 0
for i in range(1000):
	(action, reward, state) = ps.next() 
	#print action, reward, state
	total = total + reward
print "total", total / float(1000)
total = 0
model.current_state = model.state[0]
ps.e = 0.6
for i in range(10000):
	(action, reward, state) = ps.next() 
	total = total + reward
print "total", total / float(10000)
total = 0
model.current_state = model.state[0]
ps.e = 0.5
for i in range(10000):
	(action, reward, state) = ps.next() 
	total = total + reward
print "total", total/ float(10000)
total = 0
model.current_state = model.state[0]
ps.e = 0.4
for i in range(10000):
	(action, reward, state) = ps.next() 
	total = total + reward
print "total", total / float(10000)
total = 0
model.current_state = model.state[0]
ps.e = 0.3
for i in range(10000):
	(action, reward, state) = ps.next() 
	total = total + reward
print "total", total / float(10000)
total = 0
model.current_state = model.state[0]
ps.e = 0.2
for i in range(10000):
	(action, reward, state) = ps.next() 
	total = total + reward
print "total", total / float(10000)
total = 0
model.current_state = model.state[0]
ps.e = 0.1
for i in range(10000):
	(action, reward, state) = ps.next() 
	total = total + reward
print "total", total / float(10000)
