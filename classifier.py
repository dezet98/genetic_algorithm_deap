from deap import creator

from fitness import Fitnesses
from individual import Individual
from deap import tools


class Classifiers(object):
    own = "own"
    svc = "svc"
    two = "two"
    three = "three"
    four = "four"
    five = "five"
    allClassifiers = [own, svc, two, three, four, five]

    @staticmethod
    def register(name, toolbox, y=None, df=None, number_of_attributes=None):
        if name == Classifiers.own:
            toolbox.register("individual", Individual.individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.fitness_function)
        elif name == Classifiers.svc:
            toolbox.register("individual", Individual.svc, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.svc, y, df, number_of_attributes)
        elif name == Classifiers.two:
            toolbox.register("individual", Individual.two, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.two, y, df, number_of_attributes)
        elif name == Classifiers.three:
            toolbox.register("individual", Individual.three, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.three, y, df, number_of_attributes)
        elif name == Classifiers.four:
            toolbox.register("individual", Individual.four, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.four, y, df, number_of_attributes)
        elif name == Classifiers.five:
            toolbox.register("individual", Individual.five, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.five, y, df, number_of_attributes)
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(Classifiers.allClassifiers)):
            result += f"{i} - {Classifiers.allClassifiers[i]}, "
        result += "} = "
        return result
