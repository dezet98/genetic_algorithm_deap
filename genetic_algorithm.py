import math
import multiprocessing
import random

from deap import base
from deap import creator
from deap import tools

from crossover import Crossover
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection
from utils import draw_chart, save_results_to_csv, init_results_csv, print_epoch_results


class GeneticAlgorithm:
    @staticmethod
    def individual(icls):
        genome = list()
        genome.append(random.uniform(-1.5, 4))
        genome.append(random.uniform(-3, 4))

        return icls(genome)

    @staticmethod
    def fitness_function(individual_value):
        result = math.sin((individual_value[0] + individual_value[1])) + math.pow(
            (individual_value[0] - individual_value[1]), 2) - 1.5 * individual_value[0] + 2.5 * individual_value[1] + 1

        return (result,)

    @staticmethod
    def pass_operators(algorithm_params):
        algorithm_params.grade_strategy = GradeStrategy.grade_strategies[int(input(GradeStrategy.options()))]
        algorithm_params.selection = Selection.allSelection[int(input(Selection.options()))]
        algorithm_params.crossover = Crossover.allCrossover[int(input(Crossover.options()))]
        algorithm_params.mutation = Mutation.allMutation[int(input(Mutation.options()))]

    @staticmethod
    def register_operators(toolbox, algorithm_params):
        GradeStrategy(algorithm_params.grade_strategy)
        Selection(algorithm_params.selection, toolbox)
        Crossover(algorithm_params.crossover, toolbox)
        Mutation(algorithm_params.mutation, toolbox)

    @staticmethod
    def register_functions(toolbox):
        toolbox.register("individual", GeneticAlgorithm.individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", GeneticAlgorithm.fitness_function)

    @staticmethod
    def run(algorithm_params, use_global_operators, processes=1, print_results=False, save_to_csv=True,
            save_charts=True):
        global best_ind, mean, std, invalid_ind
        toolbox = base.Toolbox()
        if use_global_operators:
            GeneticAlgorithm.pass_operators(algorithm_params)
        GeneticAlgorithm.register_operators(toolbox, algorithm_params)
        GeneticAlgorithm.register_functions(toolbox)

        if __name__ == "__main__":
            pool = multiprocessing.Pool(processes=processes)
            toolbox.register("map", pool.map)

        pop = toolbox.population(n=algorithm_params.size_population)
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        g, number_elitism = 0, 1
        best_results, avg_results, std_results = [], [], []
        if save_to_csv:
            init_results_csv()
        while g < algorithm_params.number_iteration:
            g = g + 1

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(toolbox.map(toolbox.clone, offspring))

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

            if __name__ == "__main__":
                pool = multiprocessing.Pool(processes=processes)
                toolbox.register("map", pool.map)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring + list_elitism

            fits = [ind.fitness.values[0] for ind in pop]
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = tools.selBest(pop, 1)[0]

            best_results.append(best_ind.fitness.values)
            avg_results.append(mean)
            std_results.append(std)

        if save_to_csv:
            save_results_to_csv(best_ind, mean, std, algorithm_params)
        if save_charts:
            draw_chart(algorithm_params, best_results, avg_results, std_results, g, None)
        if print_results:
            print_epoch_results(pop, g, invalid_ind)
            print(algorithm_params.operators_results())
