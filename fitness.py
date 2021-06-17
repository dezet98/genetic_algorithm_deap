from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import math
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


class Fitnesses:

    @staticmethod
    def fitness_function(individual_value):
        result = math.sin((individual_value[0] + individual_value[1])) + math.pow(
            (individual_value[0] - individual_value[1]), 2) - 1.5 * individual_value[0] + 2.5 * individual_value[1] + 1

        return (result,)

    @staticmethod
    def svc(y, df, number_of_attributes, individual):
        split, df_norm = 5, Fitnesses._df_norm(number_of_attributes, individual, df)
        estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                        coef0=individual[4], random_state=1)

        return Fitnesses._train(StratifiedKFold(n_splits=split), df_norm, y, estimator, split)

    @staticmethod
    def decision_tree_classifier(y, df, number_of_attributes, individual):
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        split, df_norm = 7, Fitnesses._df_norm(number_of_attributes, individual, df)
        estimator = DecisionTreeClassifier(criterion=individual[0], splitter=individual[1], max_depth=individual[2],
                                           min_samples_split=individual[3], min_samples_leaf=individual[4],
                                           min_weight_fraction_leaf=individual[5], max_features=individual[6],
                                           random_state=1)

        return Fitnesses._train(StratifiedKFold(n_splits=split), df_norm, y, estimator, split)

    @staticmethod
    def k_neighbors_classifier(y, df, number_of_attributes, individual):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        split, df_norm = 7, Fitnesses._df_norm(number_of_attributes, individual, df)
        estimator = KNeighborsClassifier(n_neighbors=individual[0], weights=individual[1], algorithm=individual[2],
                                         leaf_size=individual[3], p=individual[4])

        return Fitnesses._train(StratifiedKFold(n_splits=split), df_norm, y, estimator, split)

    @staticmethod
    def extra_tree_classifier(y, df, number_of_attributes, individual):
        # https: // scikit - learn.org / stable / modules / generated / sklearn.tree.ExtraTreeClassifier.html  # sklearn.tree.ExtraTreeClassifier
        split, df_norm = 7, Fitnesses._df_norm(number_of_attributes, individual, df)
        estimator = ExtraTreeClassifier(criterion=individual[0], splitter=individual[1], max_depth=individual[2],
                                        min_samples_split=individual[3], min_samples_leaf=individual[4],
                                        min_weight_fraction_leaf=individual[5], max_features=individual[6],
                                        random_state=1)

        return Fitnesses._train(StratifiedKFold(n_splits=split), df_norm, y, estimator, split)

    @staticmethod
    def mlp_classifier(y, df, number_of_attributes, individual):
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
        split, df_norm = 4, Fitnesses._df_norm(number_of_attributes, individual, df)
        estimator = MLPClassifier(hidden_layer_sizes=individual[0], activation=individual[1], solver=individual[2],
                                  alpha=individual[3], max_iter=10000, random_state=1)

        return Fitnesses._train(StratifiedKFold(n_splits=split), df_norm, y, estimator, split)

    @staticmethod
    def random_forrest_classifier(y, df, number_of_attributes, individual):
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        split, df_norm = 4, Fitnesses._df_norm(number_of_attributes, individual, df)
        estimator = RandomForestClassifier(n_estimators=individual[0], criterion=individual[1], max_depth=individual[2],
                                           min_samples_split=individual[3], min_samples_leaf=individual[4],
                                           min_weight_fraction_leaf=individual[5], max_features=individual[6],
                                           max_leaf_nodes=individual[7], random_state=1)

        return Fitnesses._train(StratifiedKFold(n_splits=split), df_norm, y, estimator, split)

    @staticmethod
    def _train(cv, df_norm, y, estimator, split):
        result_sum = 0

        for train, test in cv.split(df_norm, y):
            estimator.fit(df_norm[train], y[train])
            predicted = estimator.predict(df_norm[test])
            expected = y[test]
            tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
            result = (tp + tn) / (
                    tp + fp + tn + fn)  # w oparciu o macierze pomyłek https://www.dataschool.io/simple-guide-to-confusion-matrixterminology/
            result_sum = result_sum + result  # zbieramy wyniki z poszczególnych etapów walidacji krzyżowej

        return result_sum / split,

    @staticmethod
    def _df_norm(number_of_attributes, individual, df):
        list_columns_to_drop = []  # lista cech do usuniecia
        for i in range(number_of_attributes, len(individual)):
            if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
                list_columns_to_drop.append(i - number_of_attributes)

        df_selected_features = df.drop(df.columns[list_columns_to_drop], axis=1, inplace=False)
        mms = MinMaxScaler()

        return mms.fit_transform(df_selected_features)
