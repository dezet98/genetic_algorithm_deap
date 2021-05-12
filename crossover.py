import random
from deap import tools


class Crossover:
    one_point = "one_point"
    uniform = "uniform"
    arithmetic = "_arithmetic"
    heuristic = "heuristic"
    ordered = "ordered"
    simulated_binary_bounded = "simulated_binary_bounded"

    allCrossover = [one_point, uniform, arithmetic, heuristic, ordered, simulated_binary_bounded]

    def __init__(self, name, toolbox):
        self.name = name
        if name == self.one_point:
            self._one_point(toolbox)
        elif name == self.uniform:
            self._uniform(toolbox)
        elif name == self.arithmetic:
            self._arithmetic(toolbox)
        elif name == self.heuristic:
            self._heuristic(toolbox)
        elif name == self.ordered:
            self._ordered(toolbox)
        elif name == self.simulated_binary_bounded:
            self._simulated_binary_bounded(toolbox)
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(Crossover.allCrossover)):
            result += f"{i} - {Crossover.allCrossover[i]}, "
        result += "} = "
        return result

    @staticmethod
    def _one_point(toolbox):
        toolbox.register("mate", tools.cxOnePoint)

    @staticmethod
    def _uniform(toolbox):
        toolbox.register("mate", tools.cxUniform, indpb=1.0)

    @staticmethod
    def _arithmetic(toolbox):
        toolbox.register("mate", Crossover._cx_arithmetic, indpb=1.0)

    @staticmethod
    def _heuristic(toolbox):
        toolbox.register("mate", Crossover._cx_heuristic, indpb=1.0)

    @staticmethod
    def _ordered(toolbox):
        toolbox.register("mate", tools.cxOrdered)

    @staticmethod
    def _simulated_binary_bounded(toolbox):
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=0.0, up=1.0)

    # own crossover methods
    @staticmethod
    def _cx_arithmetic(ind1, ind2, indpb):
        """Executes a arithmetic crossover that modify in place the two
        :term:`sequence` individuals. The attributes are swapped accordingto the
        *indpb* probability.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param indpb: Independent probabily for each attribute to be exchanged.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        size = min(len(ind1), len(ind2))
        for i in range(size):
            if random.random() < indpb:
                # todo
                # ind1[i], ind2[i] = ind2[i], ind1[i]
                pass

        return ind1, ind2

    @staticmethod
    def _cx_heuristic(ind1, ind2, indpb):
        """Executes a heuristic crossover that modify in place the two
        :term:`sequence` individuals. The attributes are swapped accordingto the
        *indpb* probability.

        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param indpb: Independent probabily for each attribute to be exchanged.
        :returns: A tuple of two individuals.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        size = min(len(ind1), len(ind2))
        for i in range(size):
            if random.random() < indpb:
                # todo
                # ind1[i], ind2[i] = ind2[i], ind1[i]
                pass

        return ind1, ind2
