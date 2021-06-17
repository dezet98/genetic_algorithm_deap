import random

from deap import tools

from classifier import Classifiers


class Mutation:
    gaussian = "gaussian"
    shuffle_indexes = "shuffle_indexes"
    flip_bit = "flip_bit"

    allMutation = [gaussian, shuffle_indexes, flip_bit]

    def __init__(self, name, toolbox, classifier=Classifiers.own):
        self.name = name

        if classifier == Classifiers.svc:
            toolbox.register("mutate", self.svc)
        elif classifier == Classifiers.two:
            toolbox.register("mutate", self._two)
        elif classifier == Classifiers.three:
            toolbox.register("mutate", self._three)
        elif classifier == Classifiers.four:
            toolbox.register("mutate", self._four)
        elif classifier == Classifiers.five:
            toolbox.register("mutate", self._five)
        elif name == self.gaussian:
            self._gaussian(toolbox)
        elif name == self.shuffle_indexes:
            self._shuffle_indexes(toolbox)
        elif name == self.flip_bit:
            self._flip_bit(toolbox)
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(Mutation.allMutation)):
            result += f"{i} - {Mutation.allMutation[i]}, "

        result += "} = "
        return result

    @staticmethod
    def _gaussian(toolbox):
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.5)

    @staticmethod
    def _shuffle_indexes(toolbox):
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

    @staticmethod
    def _flip_bit(toolbox):
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)

    @staticmethod
    def svc(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # kernel
            list_kernel = ["linear", "rbf", "poly", "sigmoid"]
            individual[0] = list_kernel[random.randint(0, 3)]
        elif number_parameter == 1:
            # C
            k = random.uniform(0.1, 100)
            individual[1] = k
        elif number_parameter == 2:
            # degree
            individual[2] = random.uniform(0.1, 5)
        elif number_parameter == 3:
            # gamma
            gamma = random.uniform(0.01, 1)
            individual[3] = gamma
        elif number_parameter == 4:
            # coeff
            coeff = random.uniform(0.1, 1)
            individual[2] = coeff
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _two(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # kernel
            list_kernel = ["linear", "rbf", "poly", "sigmoid"]
            individual[0] = list_kernel[random.randint(0, 3)]
        elif number_parameter == 1:
            # C
            k = random.uniform(0.1, 100)
            individual[1] = k
        elif number_parameter == 2:
            # degree
            individual[2] = random.uniform(0.1, 5)
        elif number_parameter == 3:
            # gamma
            gamma = random.uniform(0.01, 5)
            individual[3] = gamma
        elif number_parameter == 4:
            # coeff
            coeff = random.uniform(0.1, 20)
            individual[2] = coeff
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _three(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # kernel
            list_kernel = ["linear", "rbf", "poly", "sigmoid"]
            individual[0] = list_kernel[random.randint(0, 3)]
        elif number_parameter == 1:
            # C
            k = random.uniform(0.1, 100)
            individual[1] = k
        elif number_parameter == 2:
            # degree
            individual[2] = random.uniform(0.1, 5)
        elif number_parameter == 3:
            # gamma
            gamma = random.uniform(0.01, 5)
            individual[3] = gamma
        elif number_parameter == 4:
            # coeff
            coeff = random.uniform(0.1, 20)
            individual[2] = coeff
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _four(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # kernel
            list_kernel = ["linear", "rbf", "poly", "sigmoid"]
            individual[0] = list_kernel[random.randint(0, 3)]
        elif number_parameter == 1:
            # C
            k = random.uniform(0.1, 100)
            individual[1] = k
        elif number_parameter == 2:
            # degree
            individual[2] = random.uniform(0.1, 5)
        elif number_parameter == 3:
            # gamma
            gamma = random.uniform(0.01, 5)
            individual[3] = gamma
        elif number_parameter == 4:
            # coeff
            coeff = random.uniform(0.1, 20)
            individual[2] = coeff
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _five(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # kernel
            list_kernel = ["linear", "rbf", "poly", "sigmoid"]
            individual[0] = list_kernel[random.randint(0, 3)]
        elif number_parameter == 1:
            # C
            k = random.uniform(0.1, 100)
            individual[1] = k
        elif number_parameter == 2:
            # degree
            individual[2] = random.uniform(0.1, 5)
        elif number_parameter == 3:
            # gamma
            gamma = random.uniform(0.01, 5)
            individual[3] = gamma
        elif number_parameter == 4:
            # coeff
            coeff = random.uniform(0.1, 20)
            individual[2] = coeff
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0
