from deap import tools


class Mutation:
    gaussian = "gaussian"
    shuffle_indexes = "shuffle_indexes"
    flip_bit = "flip_bit"
    polynomial_bounded = "polynomial_bounded"

    allMutation = [gaussian, shuffle_indexes, flip_bit, polynomial_bounded]

    def __init__(self, name, toolbox):
        self.name = name
        if name == self.gaussian:
            self._gaussian(toolbox)
        elif name == self.shuffle_indexes:
            self._shuffle_indexes(toolbox)
        elif name == self.flip_bit:
            self._flip_bit(toolbox)
        elif name == self.polynomial_bounded:
            self._polynomial_bounded(toolbox)
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
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=1.0)

    @staticmethod
    def _shuffle_indexes(toolbox):
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0)

    @staticmethod
    def _flip_bit(toolbox):
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0)

    @staticmethod
    def _polynomial_bounded(toolbox):
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=0.0, up=1.0, indpb=1.0)
