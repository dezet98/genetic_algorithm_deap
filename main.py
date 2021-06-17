import time

from algorithm_params import AlgorithmParams
from classifier import Classifiers
from crossover import Crossover
from fitness import Fitnesses
from genetic_algorithm import GeneticAlgorithm
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection
from utils import draw_time_chart


def multiprocessing_test():
    multiprocessing_results = {'processes': [], 'time': []}

    for processes in [1, 2, 4, 8, 16]:
        t0 = time.time()

        GeneticAlgorithm.run(
            AlgorithmParams(GradeStrategy.min, Selection.best, Crossover.one_point, Mutation.shuffle_indexes,
                            size_population=90, probability_mutation=0.2, probability_crossover=0.8,
                            number_iteration=100, classifier=Classifiers.svc), processes=processes,
            use_global_operators=False,
            print_results=False, save_to_csv=False, save_charts=False)

        multiprocessing_results['time'].append(time.time() - t0)
        multiprocessing_results['processes'].append(processes)
        print('processes:', processes, 'time:', time.time() - t0)
    print(multiprocessing_results)
    draw_time_chart(multiprocessing_results['processes'], multiprocessing_results['time'])


def params_test():
    for gs in GradeStrategy.grade_strategies:
        for sel in Selection.allSelection:
            for cx in Crossover.allCrossover:
                for mut in Mutation.allMutation:
                    # try:
                    print(f"{gs} {sel} {cx} {mut}")
                    GeneticAlgorithm.run(
                        AlgorithmParams(gs, sel, cx, mut, size_population=100, probability_mutation=0.2,
                                        probability_crossover=0.8, number_iteration=150), False)
                    # except TypeError as err:
                    #     print(f"TypeError: \n{err}")


def single_test():
    GeneticAlgorithm.run(
        AlgorithmParams(GradeStrategy.min, Selection.best, Crossover.one_point, Mutation.shuffle_indexes,
                        size_population=90, probability_mutation=0.2, probability_crossover=0.8,
                        number_iteration=100, classifier=Classifiers.svc), processes=1,
        use_global_operators=False,
        print_results=True, save_to_csv=True, save_charts=True)


def own_test():
    GeneticAlgorithm.run(
        AlgorithmParams(GradeStrategy.min, Selection.best, Crossover.one_point, Mutation.gaussian, size_population=70,
                        probability_mutation=0.2, probability_crossover=0.8,
                        number_iteration=70),
        processes=1, use_global_operators=True,
        print_results=True, save_to_csv=True, save_charts=True)


if __name__ == '__main__':
    # multiprocessing_test()
    # params_test()
    # single_test()
    # own_test()
    single_test()
