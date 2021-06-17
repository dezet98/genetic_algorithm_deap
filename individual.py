import random


class Individual:

    @staticmethod
    def individual(icls):
        genome = list()
        genome.append(random.uniform(-1.5, 4))
        genome.append(random.uniform(-3, 4))

        return icls(genome)

    @staticmethod
    def svc(number_features, icls):
        genome = list()
        # kernel
        list_kernel = ["linear", "rbf", "poly", "sigmoid"]
        genome.append(list_kernel[random.randint(0, 3)])
        # c
        k = random.uniform(0.1, 100)
        genome.append(k)
        # degree
        genome.append(random.uniform(0.1, 5))
        # gamma
        gamma = random.uniform(0.001, 5)
        genome.append(gamma)
        # coeff
        coeff = random.uniform(0.01, 10)
        genome.append(coeff)

        for i in range(0, number_features):
            genome.append(random.randint(0, 1))

        return icls(genome)

    @staticmethod
    def two(number_features, icls):
        genome = list()

        # todo

        return icls(genome)

    @staticmethod
    def three(number_features, icls):
        genome = list()

        # todo

        return icls(genome)

    @staticmethod
    def four(number_features, icls):
        genome = list()

        # todo

        return icls(genome)

    @staticmethod
    def five(number_features, icls):
        genome = list()

        # todo

        return icls(genome)
