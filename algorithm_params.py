from crossover import Crossover
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection


class AlgorithmParams:
    size_population = 100
    probability_mutation = 0.2
    probability_crossover = 0.8
    number_iteration = 100

    def __init__(self, grade_strategy=GradeStrategy.min, selection=Selection.best, crossover=Crossover.one_point,
                 mutation=Mutation.gaussian):
        self.grade_strategy = grade_strategy
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def operators_results(self):
        return f"-- Grade_strategy: {self.grade_strategy}, Selection: {self.selection}, Crossover: {self.crossover}, Mutation: {self.mutation} -- "
