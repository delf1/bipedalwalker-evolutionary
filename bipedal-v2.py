import random
import gym
import multiprocessing
import csv
import sys

from deap import base
from deap import creator
from deap import tools

env = gym.make('BipedalWalker-v2')

episodes = 0


#START CODE TO RUN ALREADY EVOLVED ROBOT
def performAction(observation, ind):
    #0 is left hip
    #1 is left knee
    #2 is right hip
    #3 is right knee

    individual = []

    if ind == 1:
        #1000 GENS, 200 pop, this one is stable
        individual = [1.7435976016832306, 0.0, 0.49074562542871014, 0.6352274461669776, -1.005563298234311, -3.3713317539451406, 0.0, 0.0, -0.885567670756612, -4.908833036088131, 0.14998613771151081, -3.4144352161466207, 2.6198401145873067, 0.0, 4.203808832337277, 4.161590214543905, 0.0, 1.2415183929252551, -3.019773327541343, 0.0, -4.634478828921555, 3.6444713241415094, -2.6294673164240345, 1.0, 0.0, 1.680682138121231, 4.730149180446189, 0.0, 0.0, -0.049606871412364306, -2.2221906086308216, -2.9115491446009845, -4.1289211393818075, 0.0, -4.970231848176791, -1.904240599700222, 0.0, -1.33316766835296, -2.7519161957883176, -3.119115642186091, 0.0, -0.5668164116009375, 0.0, 1.0, 4.963303933648977, 1.5705754458336614, -0.7609444791313669, 0.0, -2.889782882612372, 0.5303671399428911, 1.0, 4.476066154706221, 0.0, 0.0, 1.8499606854898287, 0.0]
    elif ind == 2:
        #1000 GENS, 1000 Pop, this one is faster but less stable than first
        individual = [0.0, 6.853330139043791, 0.0, 1.0, 0.0, 1.0919664926147306, -0.6394341928997598, 0.0, -4.670966834542329, 0.0, -1.7906787143810057, -6.601356963074167, 1.2265637139485115, 3.256934511937395, 5.931076094106739, 0.0, -1.1181944057449265, 0.0, 0.7340980178816423, 7.4913279435102105, 0.0, 1.0, -5.3796647399105035, 0.0, 0.0, 0.0, 8.328930327251282, 0.0, -7.39850553764694, -5.657798958867948, -4.8270490638356645, 0.061187480210884715, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -2.699382732614424, 1.0, 5.130195858483873, 1.0727115097515636, 8.208099513827992, 4.010310032100081, -6.649968760481864, 0.0, 9.014779419971529, 0.0, 2.5360645793481833, 0.6629500887708204, 0.0, 6.902869527663498, 5.286965370093091, -3.7098507311816835, 8.502104327926016, -5.683207107777131]
    else:
        #1000 GENS, 200 pop, this one is faster than second but much less stable
        individual = [1.0, -2.2856756127189453, 0.0, -3.288920041732508, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -9.442043927263718, -7.942950617353082, 1.0, 0.5402476803343745, -0.3494702211859355, -5.861565511961526, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -8.8774384607399, 1.0, 4.9665398337905415, -9.144001591714776, 0.0, 3.314163797556361, 1.0, 4.91431707083928, 4.221425366664548, 0.0, -6.157720804498597, 3.3496786403703727, -0.9345987644902216, 0.0, -9.335532853397703, 0.0, 0.0, 0.0, 0.0, -8.095742562992807, -6.744336997442453, -7.148868800307728, 6.953580338441366, 0.0, 3.4698680938191977, -5.96435334704654, 7.708974300054159, 1.0, -8.96569939471107, 9.118168417598088, 0.0, -9.789149042808807, 6.242560298147609, 1.0]

    a1 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[0:14], observation[0:14]), 0)
    a2 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[14:28], observation[0:14]), 0)
    a3 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[28:42], observation[0:14]), 0)
    a4 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[42:56], observation[0:14]), 0)

    return [a1,a2,a3,a4]

def runEnv(ind):
    while True:
        observation = env.reset()
        tot_reward = 0
        done = False
        t = 0

        while not done:
            env.render()

            action = performAction(observation, ind)
        
            observation, reward, done, info = env.step(action)
            tot_reward = tot_reward + reward
            t = t + 1

            if done:
                print("Episode finished after {} timesteps with total reward {}".format(t+1, tot_reward))
                break
    
#END CODE TO RUN ALREADY EVOLVED ROBOT

#START CODE TO TRAIN
def calculateAction(individual, observation):
    #0 is left hip
    #1 is left knee
    #2 is right hip
    #3 is right knee

    a1 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[0:14], observation[0:14]), 0)
    a2 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[14:28], observation[0:14]), 0)
    a3 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[28:42], observation[0:14]), 0)
    a4 = reduce((lambda x, y: x + y[0] * y[1]) ,zip(individual[42:56], observation[0:14]), 0)
    return [a1,a2,a3,a4]

# the goal ('fitness') function to be maximized
def evalFitness(individual):
    observation = env.reset()
    tot_reward = 0
    done = False
    t = 0

    while not done:

        action = calculateAction(individual, observation)
        
        observation, reward, done, info = env.step(action)
        tot_reward = tot_reward + reward
        t = t + 1

        if done:
            break
    global episodes
    episodes = episodes + 1
    return tot_reward,

def useGenetic(multi = False):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # writer = csv.writer(open('bipedal-v2-stats.csv', 'wb'))
    # writer.writerow(['gen', 'min', 'max', 'avg', 'std'])

    toolbox = base.Toolbox()
    
    if(multi == True):
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

    # Attribute generator 
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to floats sampled uniformly
    #                      from the range (-10,10)

    toolbox.register("attr_float", random.uniform, -10, 10)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of 56 'attr_float' elements ('genes')
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_float, 56)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", evalFitness)

    # register the crossover operator
    toolbox.register("mate", tools.cxUniform, indpb=0.125)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed()

    # create an initial population of 200 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=200)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    max_fit = max(fits)
    # Begin the evolution
    while g < 1000 and max_fit < 300:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        best_ind = tools.selBest(pop, 1)[0]
        print "Best individual: " + str(list(best_ind))
        # f = open('best_indivdual_bipedal_v2', 'w')
        # f.write((str(list(best_ind))))
        # f.close()

        # writer.writerow([g, min(fits), max(fits), mean, std])

        max_fit = max(fits)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print "Number of Episodes: " + str(episodes)

#----------

def main():
    if len(sys.argv) == 1:
        runEnv(3)
    else:
        if sys.argv[1] == '1':
            runEnv(1)
        elif sys.argv[1] == '2':
            runEnv(2)
        elif sys.argv[1] == '3':
            runEnv(3)
        elif sys.argv[1] == 'train':
            if len(sys.argv) == 2:
                useGenetic(False)
            elif sys.argv[2] == 'multi':
                useGenetic(True)
            else:
                print "bad argument, use: python bipedal-v2.py train multi"
        else:
            print "bad argument, use: python bipedal-v2.py {1, 2 or 3}"

if __name__ == "__main__":
    main()