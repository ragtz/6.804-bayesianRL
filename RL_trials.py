import sys
from PrioritizedSweeping import *
from PrioritizedSweepingPolicy import *
from PrioritizedSweepingHeuristics import *
from PrioritizedQLearning import *
from QLearning import *
from QLearn import *
from ChainModel import *
from ChainModel2 import *
from LoopModel import *
from BayesPrioritizedSweeping import *

def get_learner(algorithm, model):
    if algorithm == "QLearning":
        return QLearning(model)
    elif algorithm == "PrioritizedSweeping":
        return PrioritizedSweeping(model)
    elif algorithm == "PrioritizedSweepingPolicy":
        return PrioritizedSweepingPolicy(model)
    elif algorithm == "PrioritizedSweepingHeuristics":
        return PrioritizedSweepingHeuristics(model)
    elif algorithm == "QLearn":
        return QLearn(model)
    elif algorithm == "PrioritizedQLearning":
        return PrioritizedQLearning(model)
    elif algorithm == "BayesDP":
        return BayesPrioritizedSweeping(model)
    else:
        raise Exception(algorithm + " not found")

def get_model(model_name):
    if model_name == "Chain":
        return ChainModel()
    elif model_name == "Loop":
        return LoopModel()
    elif model_name == "SpecialLoopModel":
        return SpecialLoopModel()
    elif model_name == "Chain2":
        return ChainModel2()

def run_trials(num_trials, num_phases, num_steps, algorithm = "QLearning", model_name = "Chain"):
    learners = []
    for i in range(num_trials):
        m = get_model(model_name)
        #learners.append(QLearn(m,m.actions))
        learners.append(get_learner(algorithm, m))

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
        model_name = "Chain"
        if len(sys.argv) == 6:
            model_name = sys.argv[5]
        avgs = run_trials(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], model_name)
        for (i,avg) in enumerate(avgs):
            print "Phase " + str(i+1) + ": " + str(avg)
    else:
        print "Invalid input."
