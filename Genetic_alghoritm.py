from deap import base
from deap import creator
from deap import tools

import random
import time
import matplotlib.pyplot as plt


def getParams():
    size_population = int(input("Size population: "))
    number_iteration = int(input("Iteration number: "))
    selection = int(input("Choose selection (write a number):\n 1. Tournament\n 2. Stochastic Universal Sampling\n 3. "
                          "Lexicase\n 4. Roulette\n 5. Random\n 6. Best\n 7. Worst\n"))
    if int(selection) == 1:
        sel_param = int(input("Tournament size: "))
    else:
        sel_param = None

    crossover = int(
        input("Choose crossover (write a number):\n 1. One Point\n 2. Two Point\n 3. Uniform\n 4. Blend\n"))
    prob_crossover = float(input("Crossover probability: "))
    if int(crossover) == 4:
        alpha = float(input("Alpha for the blend crossover: "))
    else:
        alpha = None

    mutation = int(input(
        "Choose mutation (write a number):\n 1. Gaussian\n 2. Shuffle Indexes\n 3. Flip Bit\n"))
    prob_mutation = float(input("Mutation probability: "))
    if int(mutation) == 1:
        mut_param = int(input("Mu for the gaussian addition mutation: "))
        sigma = float(input("Sigma for the gaussian addition mutation: "))
    else:
        mut_param = None
        sigma = None

    return size_population, number_iteration, selection, sel_param, crossover, prob_crossover, alpha, mutation, prob_mutation, mut_param, sigma


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))
    return icls(genome)


def fitnessFunction(ind):
    return pow((ind[0] + 2 * ind[1] - 7), 2) + pow((2 * ind[0] + ind[1] - 5), 2)


def setSelection(toolbox, selection, param):
    if selection == 1:
        toolbox.register("select", tools.selTournament, tournsize=param)
    elif selection == 2:
        toolbox.register("select", tools.selStochasticUniversalSampling)
    elif selection == 3:
        toolbox.register("select", tools.selLexicase)
    elif selection == 4:
        toolbox.register("select", tools.selRoulette)
    elif selection == 5:
        toolbox.register("select", tools.selRandom)
    elif selection == 6:
        toolbox.register("select", tools.selBest)
    elif selection == 7:
        toolbox.register("select", tools.selWorst)


def setCrossover(toolbox, crossover, prob_crossover, alpha):
    if crossover == 1:
        toolbox.register("mate", tools.cxOnePoint)
    elif crossover == 2:
        toolbox.register("mate", tools.cxTwoPoint)
    elif crossover == 3:
        toolbox.register("mate", tools.cxUniform, indpb=prob_crossover)
    elif crossover == 4:
        toolbox.register("mate", tools.cxBlend, alpha=alpha)


def setMutation(toolbox, mutation, prob_mutation, param, sigma):
    if mutation == 1:
        toolbox.register("mutate", tools.mutGaussian, mu=param, sigma=sigma, indpb=prob_mutation)
    elif mutation == 2:
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=prob_mutation)
    elif mutation == 3:
        toolbox.register("mutate", tools.mutFlipBit, indpb=prob_mutation)


def getToolbox(selection, sel_param, crossover, prob_crossover, alpha, mutation, prob_mutation, mut_param, sigma):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitnessFunction)

    setSelection(toolbox, selection, sel_param)
    setCrossover(toolbox, crossover, prob_crossover, alpha)
    setMutation(toolbox, mutation, prob_mutation, mut_param, sigma)

    return toolbox


def showPlot(data, ylabel):
    plt.figure(figsize=(16, 5))
    plt.plot(data)
    plt.xlabel('Generation')
    plt.ylabel(ylabel)
    plt.show()


def setResults(pop, best_ind, all_min, all_mean, all_std):
    fits = [ind.fitness.values[0] for ind in pop]
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print(" Min %s" % min(fits))
    print(" Max %s" % max(fits))
    print(" Avg %s" % mean)
    print(" Std %s" % std)
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    all_min.append(best_ind.fitness.values)
    all_mean.append(mean)
    all_std.append(std)


def getFinalResults(best_ind, best_solution, all_min, all_mean, all_std):
    print("Best individual is %s" % best_ind)
    print("Best solution is %s" % best_solution)

    showPlot(all_min, "Value of the fitness function")
    showPlot(all_mean, "Mean value of the fitness function")
    showPlot(all_std, "Standard deviation")


def minimalizeFunction():
    size_population, number_iteration, selection, sel_param, crossover, prob_crossover, alpha, mutation, prob_mutation, mut_param, sigma = getParams()

    best_solution = None
    best_individual = None
    all_min = []
    all_mean = []
    all_std = []

    toolbox = getToolbox(selection, sel_param, crossover, prob_crossover, alpha, mutation, prob_mutation, mut_param,
                         sigma)
    pop = toolbox.population(n=size_population)
    g = 0

    start = time.time()
    while g < number_iteration:
        g = g + 1
        print("-- Generation %i --" % g)

        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < prob_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < prob_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        print(" Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring

        best_ind = tools.selBest(pop, 1)[0]
        setResults(pop, best_ind, all_min, all_mean, all_std)
        if best_solution is None or best_solution > best_ind.fitness.values:
            best_solution = best_ind.fitness.values
            best_individual = best_ind

    end = time.time()
    print("\n\nTotal time:", end - start, " sec")
    print("-- End of (successful) evolution --")
    getFinalResults(best_individual, best_solution, all_min, all_mean, all_std)


minimalizeFunction()
