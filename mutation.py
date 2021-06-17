import random

from deap import tools

from classifier import Classifiers


class Mutation:
    gaussian = "gaussian"
    shuffle_indexes = "shuffle_indexes"
    flip_bit = "flip_bit"

    allMutation = [gaussian, shuffle_indexes, flip_bit]

    def __init__(self, name, toolbox, classifier=Classifiers.own):
        if classifier != Classifiers.own:
            self.name = '-'
        else:
            self.name = name

        if classifier == Classifiers.svc:
            toolbox.register("mutate", self.svc)
        elif classifier == Classifiers.decision_tree_classifier:
            toolbox.register("mutate", self._decision_tree_classifier)
        elif classifier == Classifiers.k_neighbors_classifier:
            toolbox.register("mutate", self._k_neighbors_classifier)
        elif classifier == Classifiers.extra_tree_classifier:
            toolbox.register("mutate", self._extra_tree_classifier)
        elif classifier == Classifiers.mlp_classifier:
            toolbox.register("mutate", self._mlp_classifier)
        elif classifier == Classifiers.random_forrest_classifier:
            toolbox.register("mutate", self._random_forrest_classifier)
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
            k = random.uniform(0.1, 80)
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
    def _decision_tree_classifier(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # criterion
            list_criterion = ["gini", "entropy"]
            individual[0] = list_criterion[random.randint(0, 1)]
        elif number_parameter == 1:
            # splitter
            list_splitter = ["best", "random"]
            individual[1] = list_splitter[random.randint(0, 1)]
        elif number_parameter == 2:
            # max_depth
            max_depth = random.randint(2, 10)
            individual[2] = max_depth
        elif number_parameter == 3:
            # min_samples_split
            min_samples_split = random.randint(2, 10)
            individual[3] = min_samples_split
        elif number_parameter == 4:
            # min_samples_leaf
            min_samples_leaf = random.randint(1, 10)
            individual[4] = min_samples_leaf
        elif number_parameter == 5:
            # min_weight_fraction_leaf
            min_weight_fraction_leaf = random.uniform(0, 0.5)
            individual[5] = min_weight_fraction_leaf
        elif number_parameter == 6:
            # max_features
            list_max_features = ["auto", "sqrt", "log2"]
            individual[6] = list_max_features[random.randint(0, 2)]
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _k_neighbors_classifier(individual):
        number_parameter = random.randint(0, len(individual) - 1)

        if number_parameter == 0:
            # n_neighbors
            n_neighbors = random.randint(1, 10)
            individual[0] = n_neighbors
        elif number_parameter == 1:
            # weights
            list_weights = ["uniform", "distance"]
            individual[1] = list_weights[random.randint(0, 1)]
        elif number_parameter == 2:
            # algorithm
            list_algorithm = ["auto", "ball_tree", "kd_tree", "brute"]
            individual[2] = list_algorithm[random.randint(0, 3)]
        elif number_parameter == 3:
            # leaf_size
            leaf_size = random.randint(1, 100)
            individual[3] = leaf_size
        elif number_parameter == 4:
            # p
            p = random.randint(1, 5)
            individual[4] = p
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _extra_tree_classifier(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # criterion
            list_criterion = ["gini", "entropy"]
            individual[0] = list_criterion[random.randint(0, 1)]
        elif number_parameter == 1:
            # splitter
            list_splitter = ["random", "best"]
            individual[1] = list_splitter[random.randint(0, 1)]
        elif number_parameter == 2:
            # max_depth
            max_depth = random.randint(2, 10)
            individual[2] = max_depth
        elif number_parameter == 3:
            # min_samples_split
            min_samples_split = random.randint(2, 10)
            individual[3] = min_samples_split
        elif number_parameter == 4:
            # min_samples_leaf
            min_samples_leaf = random.randint(1, 10)
            individual[4] = min_samples_leaf
        elif number_parameter == 5:
            # min_weight_fraction_leaf
            min_weight_fraction_leaf = random.uniform(0, 0.5)
            individual[5] = min_weight_fraction_leaf
        elif number_parameter == 6:
            # max_features
            list_max_features = ["auto", "sqrt", "log2"]
            individual[6] = list_max_features[random.randint(0, 2)]
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _mlp_classifier(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # hidden_layer_sizes
            hidden_layer_sizes = random.randint(1, 200)
            individual[0] = hidden_layer_sizes
        elif number_parameter == 1:
            # algorithm
            list_activation = ["identity", "logistic", "tanh", "relu"]
            individual[1] = list_activation[random.randint(0, 3)]
        elif number_parameter == 2:
            # solver
            list_solver = ["lbfgs", "sgd", "adam"]
            individual[2] = list_solver[random.randint(0, 2)]
        elif number_parameter == 3:
            # alpha
            alpha = random.random()
            individual[3] = alpha
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0

    @staticmethod
    def _random_forrest_classifier(individual):
        number_parameter = random.randint(0, len(individual) - 1)
        if number_parameter == 0:
            # n_estimators
            n_estimators = random.randint(10, 100)
            individual[0] = n_estimators
        elif number_parameter == 1:
            # criterion
            list_criterion = ["gini", "entropy"]
            individual[1] = list_criterion[random.randint(0, 1)]
        elif number_parameter == 2:
            # max_depth
            max_depth = random.randint(1, 100)
            individual[2] = max_depth
        elif number_parameter == 3:
            # min_samples_split
            min_samples_split = random.randint(2, 50)
            individual[3] = min_samples_split
        elif number_parameter == 4:
            # min_samples_leaf
            min_samples_leaf = random.randint(1, 50)
            individual[4] = min_samples_leaf
        elif number_parameter == 5:
            # min_weight_fraction_leaf
            min_weight_fraction_leaf = random.uniform(0, .5)
            individual[5] = min_weight_fraction_leaf
        elif number_parameter == 6:
            # max_features
            list_max_features = ["auto", "log2"]
            individual[6] = list_max_features[random.randint(0, 1)]
        elif number_parameter == 7:
            # max_leaf_nodes
            max_leaf_nodes = random.randint(2, 50)
            individual[7] = max_leaf_nodes
        else:
            # genetyczna selekcja cech
            if individual[number_parameter] == 0:
                individual[number_parameter] = 1
            else:
                individual[number_parameter] = 0
