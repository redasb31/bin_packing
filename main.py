import time
from lib.classes import Bin, Item
from lib.utils import *
from lib.algorithms import *
import argparse

def test_branch_and_bound(items, bin_capacity):

    # first fit

    t0 = time.time()
    bins = first_fit(items, bin_capacity)
    t0 = round(time.time()-t0, 6)

    # branch and bound with stack
    # t1 = time.time()
    # bins1 = stack_branch_and_bound(items, bins, bin_capacity)
    # t1 = round(time.time() - t1, 6)

    # plot_1_bin(bins1, t1,  "Branch and Bound", len(items))

    # recursive branch and bound
    # t2 = time.time()
    # bins2 = recursive_branch_and_bound(items, [], bins)
    # t2 = round(time.time() - t2, 6)

    # branch and bound with stack using dynamic programming
    t3 = time.time()
    bins3 = dynamic_branch_and_bound(items, bins, bin_capacity)
    t3 = round(time.time()- t3, 6)

    plot_1_bin(bins3, t3,  "dynamic Branch and Bound", len(items), f"dynamic Branch and Bound - {len(items)} items")

def test_heuristics():
    algorithms = ["first_fit", "last_fit", "best_fit", "worst_fit", "next_fit"]
    all_bins = []
    times = []

    items.sort()
    for algo in algorithms:
        t0 = time.time()
        bins = eval(algo)(items, bin_capacity)
        t = round(time.time() - t0, 6)
        times.append(t)
        all_bins.append(bins)
        plot_1_bin(bins, t, algo, len(items))


    # Plotting results
    # plot_5_bins(
    #     all_bins,
    #     times,
    #     algorithms
    #     )
    
def test_taboo_search():
    bins = first_fit(items,bin_capacity)
    nb_iterations = 2000
    taboo_list_size=10

    t0 = time.time()
    solution = taboo_search(bins, nb_iterations, taboo_list_size)
    t = round(time.time() - t0, 6)
    
    plot_1_bin(solution, t, "Taboo Search", f"Taboo Search - {nb_items} items - {nb_iterations} iterations - {taboo_list_size} taboo list size")

def test_dispersed_search(items, bin_capacity):

    num_subspaces = 20
    population_size = 100
    num_generations = 250
    crossover_rate = 0.7
    mutation_rate = 0.03

    t0 = time.time()
    solution = dispersed_genetic_algorithm(items, bin_capacity, num_subspaces, population_size, num_generations,
        crossover_rate, mutation_rate)
    # print(solution)
    t = round(time.time() - t0,6)
    plot_1_bin(solution[1], t, "Dispersed Genetic Algorithm",len(items), f"Dispersed Genetic Algorithm - {nb_items} items - {num_subspaces} subspaces - {population_size} population size - {num_generations} generations - {crossover_rate} crossover rate - {mutation_rate} mutation rate")

def test_hybridation(items, bin_capacity):

    population_size = 500
    num_subspaces = population_size // 10
    num_generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.2
    grasp_iterations = 1
    grasp_alpha = 0.2

    t0 = time.time()
    solution = hybrid_algorithm(items, bin_capacity, num_subspaces, population_size, num_generations,
        crossover_rate, mutation_rate, grasp_iterations, grasp_alpha)
    
    t = round(time.time() - t0)
    plot_1_bin(solution[1], t, "Hybrid Algorithm",len(items), f"Hybrid Algorithm - {nb_items} items - {num_subspaces} subspaces - grasp iterations = {grasp_iterations} - grasp alpha = {grasp_alpha} - {population_size} population size - {num_generations} generations - {crossover_rate} crossover rate - {mutation_rate} mutation rate")

def fitnesses_over_population_size_num_generations(items, bin_capacity,):
    if False:
        population_size_interval = list(range(100, 500, 50))
        num_generations_interval = list(range(50, 500, 50))
        crossover_rate = 0.8
        mutation_rate = 0.2
        grasp_iterations = 1
        grasp_alpha = 0.5

        # plot fitness change over populations, num_generations, grasp_iterations
        fitnesses=[]
        for population_size in population_size_interval:
            num_subspaces = population_size // 10
            for num_generations in num_generations_interval:
                t0 = time.time()
                fitness, _ = hybrid_algorithm(items, bin_capacity, num_subspaces, population_size, num_generations,
                    crossover_rate, mutation_rate, grasp_iterations, grasp_alpha)

                fitnesses.append(fitness)


                t = round(time.time() - t0)
                print(population_size, num_generations, fitness)

        x = population_size_interval
        y = num_generations_interval
        z = fitnesses
        
        # save fitnesses to file
        with open("data/fitnesses_over_population_size_num_generations.py", "w") as f:
            f.write(f"x = {str(x)}\n")
            f.write(f"y = {str(y)}\n")
            f.write(f"z = {str(z)}\n")

    else:
        # load data from file
        from data.fitnesses_over_population_size_num_generations import x, y, z
    assert len(x) *len(y) == len(z)
    plot_heatmap(x, y, z, "population size", "num generations", "fitness", "fitness over population size and num generations")

def fitnesses_over_grasp_iterations_grasp_alpha(items, bin_capacity,):
    if False:
        population_size = 500
        num_subspaces = population_size // 10
        num_generations = 100
        crossover_rate = 0.8
        mutation_rate = 0.2
        grasp_iterations_interval = list(range(1, 10, 1))
        grasp_alpha_interval = [0.1, 0.3, 0.5, 0.7, 0.9]

        # plot fitness change over populations, num_generations, grasp_iterations
        fitnesses=[]
        for grasp_iterations in grasp_iterations_interval:
            for grasp_alpha in grasp_alpha_interval:
                t0 = time.time()
                fitness, _ = hybrid_algorithm(items, bin_capacity, num_subspaces, population_size, num_generations,
                    crossover_rate, mutation_rate, grasp_iterations, grasp_alpha)

                fitnesses.append(fitness)


                t = round(time.time() - t0)
                print(population_size, num_generations, fitness)

        x = grasp_iterations_interval
        y = grasp_alpha_interval
        z = fitnesses
        
        # save fitnesses to file
        with open("data/fitnesses_over_grasp_iterations_grasp_alpha.py", "w") as f:
            f.write(f"x = {str(x)}\n")
            f.write(f"y = {str(y)}\n")
            f.write(f"z = {str(z)}\n")

    else:
        # load data from file
        from data.fitnesses_over_grasp_iterations_grasp_alpha import x, y, z


    assert len(x) *len(y) == len(z)
    plot_heatmap(x, y, z, "grasp_iterations", "grasp_alpha", "fitness", "fitness over grasp_iterations and grasp_alpha")


if __name__ == "__main__":

    # command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb-items', type=int, default=100, help='number of items')
    parser.add_argument('--max-size', type=int, default=1000, help='maximum size')
    parser.add_argument('--bin-capacity', type=int, default=1000, help='bin capacity')
    args = parser.parse_args()

    nb_items = args.nb_items
    max_size = args.max_size
    bin_capacity = args.bin_capacity
    
    # print(f"Generating {nb_items} items with max size = {max_size}, bin capacity = {bin_capacity} .")
    # items = genereate_items(nb_items, max_size)
    

    dataset = open("./BPP_50_50_0.1_0.7_0.txt", "r").read().strip().split("\n")
    nb_items = int(dataset[0])
    bin_capacity = int(dataset[1])

    # delete first two lines
    del dataset[0]
    del dataset[0]

    random.shuffle(dataset)

    items = []
    for i in range(nb_items):
        items.append(Item(int(dataset[i])))
    print(f"Generating {nb_items} items with max size = {max_size}, bin capacity = {bin_capacity} .")

    # test_branch_and_bound(items, bin_capacity)
    # test_heuristics()
    # test_taboo_search()
    
    # for i in range(50):
    #     test_hybridation(items, bin_capacity)

    # fine_tuning(items, bin_capacity)
    # fitnesses_over_population_size_num_generations(items, bin_capacity)
    fitnesses_over_grasp_iterations_grasp_alpha(items, bin_capacity)
    #test_hybridation(items, bin_capacity)



    






            