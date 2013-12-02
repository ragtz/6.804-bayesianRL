import sys
from PrioritizedSweeping import *
from QLearning import *
from ChainModel import *
from LoopModel import *

def run_trials(num_trails, num_phases, num_steps):
    learners = []
    for i in range(num_trails):
        learners.append(QLearning(SlipperyChainModel(),0.01,0.05))
    
    phase_avgs = []
    for i in range(num_phases):
        totals = []
        for learner in learners:
            total = 0
            for j in range(num_steps):
                (action, reward, next_state) = learner.next()
                total += reward
            totals.append(total)
            
        phase_avgs.append(sum(totals)/(1.0*len(totals)))
        
        for learner in learners:
            learner.model.reset()
            
    return phase_avgs
            
            
if __name__ == '__main__':
    if len(sys.argv) >= 4:
        avgs = run_trials(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        for (i,avg) in enumerate(avgs):
            print "Phase " + str(i+1) + ": " + str(avg)
    else:
        print "Invalid input."
        
