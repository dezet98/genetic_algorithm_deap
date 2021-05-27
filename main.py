import multiprocessing
import random
from deap import base
from deap import creator
from deap import tools
from crossover import Crossover
from algorithm_params import AlgorithmParams
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection
from utils import print_epoch_results, draw_chart


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))

    return icls(genome)


def fitness_function(individual_value):
    # todo change function
    result = (individual_value[0] + 2 * individual_value[1] - 7) ** 2 + (
            2 * individual_value[0] + individual_value[1] - 5) ** 2

    return (result,)


def pass_operators(algorithm_params):
    algorithm_params.grade_strategy = GradeStrategy.grade_strategies[int(input(GradeStrategy.options()))]
    algorithm_params.selection = Selection.allSelection[int(input(Selection.options()))]
    algorithm_params.crossover = Crossover.allCrossover[int(input(Crossover.options()))]
    algorithm_params.mutation = Mutation.allMutation[int(input(Mutation.options()))]


def register_operators(toolbox, algorithm_params):
    GradeStrategy(algorithm_params.grade_strategy)
    Selection(algorithm_params.selection, toolbox)
    Crossover(algorithm_params.crossover, toolbox)
    Mutation(algorithm_params.mutation, toolbox)


def register_functions(toolbox):
    toolbox.register("individual", individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function)


def genetic_algorithm(algorithm_params, use_global_operators):
    toolbox = base.Toolbox()
    if use_global_operators:
        pass_operators(algorithm_params)
    register_operators(toolbox, algorithm_params)
    register_functions(toolbox)

    # multiprocessing
    # if __name__ == "__main__":
    #     pool = multiprocessing.Pool(processes=4)
    #     toolbox.register("map", pool.map)

    pop = toolbox.population(n=algorithm_params.size_population)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0
    number_elitism = 1
    best_results, avg_results, std_results = [], [], []
    while g < algorithm_params.number_iteration:
        g = g + 1

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        list_elitism = []
        for x in range(0, number_elitism):
            list_elitism.append(tools.selBest(pop, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < algorithm_params.probability_crossover:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < algorithm_params.probability_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # multiprocessing
        # if __name__ == "__main__":
        #     pool = multiprocessing.Pool(processes=4)
        #     toolbox.register("map", pool.map)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring + list_elitism

        # print_epoch_results(pop, g, invalid_ind)

        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        best_ind = tools.selBest(pop, 1)[0]

        best_results.append(best_ind.fitness.values[0])
        avg_results.append(mean)
        std_results.append(std)

    # best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    # todo
    draw_chart(algorithm_params, best_results, avg_results, std_results, g, None)
    # print(best_results, avg_results, std_results, sep="\n\n")
    # print(algorithm_params.operators_results())


if __name__ == '__main__':
    # genetic_algorithm(AlgorithmParams(GradeStrategy.min, Selection.best, Crossover.one_point, Mutation.gaussian),
    # False)

    for gs in GradeStrategy.grade_strategies:
        for sel in Selection.allSelection:
            for cx in Crossover.allCrossover:
                for mut in Mutation.allMutation:
                    # try:
                    print(f"{gs} {sel} {cx} {mut}")
                    genetic_algorithm(AlgorithmParams(gs, sel, cx, mut), False)
                    # except TypeError as err:
                    #     print(f"TypeError: \n{err}")
