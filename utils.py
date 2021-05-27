from deap import tools
import matplotlib.pyplot as plt


def print_epoch_results(pop, g, invalid_ind):
    print("-- Generation %i --" % g)
    if invalid_ind is not None:
        print(" Evaluated %i individuals" % len(invalid_ind))

    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print(" Min %s" % min(fits))
    print(" Max %s" % max(fits))
    print(" Avg %s" % mean)
    print(" Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


def draw_chart(algorithm_params, best_results, avg_results, std_results, generation, invalid_ind):
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(algorithm_params.operators_results(), fontsize=16)

    save_best(fig.add_subplot(221), best_results, generation)
    save_average(fig.add_subplot(222), avg_results, generation)
    save_std(fig.add_subplot(223), std_results, generation)

    file_name = "data/" + algorithm_params.file_path() + ".png"

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    # plt.show()


def save_best(ax, best_results, generation):
    ax.scatter(list(range(1, generation + 1)), best_results, s=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best value')


def save_average(ax, avg_results, generation):
    ax.scatter(list(range(1, generation + 1)), avg_results, s=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average value')


def save_std(ax, std_results, generation):
    ax.scatter(list(range(1, generation + 1)), std_results, s=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('The standard deviation')
