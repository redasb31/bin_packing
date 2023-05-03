import random
import matplotlib as plt
import time
import itertools
from lib.classes import Bin, Item
from lib.utils import plot_5_bins, genereate_items,plot_1_bin, plot_2_bins
from lib.algorithms import first_fit, recursive_branch_and_bound, stack_branch_and_bound, dynamic_branch_and_bound, last_fit, best_fit, worst_fit, next_fit, taboo_search
import argparse

def test_heuristics():
    algorithms = ["first_fit", "last_fit", "best_fit", "worst_fit", "next_fit"]
    all_bins = []
    times = []

    items.sort()
    for algo in algorithms:
        t0 = time.time()
        bins = eval(algo)(items, bin_capacity)
        t1 = time.time()
        times.append(round(t1 - t0, 6))
        all_bins.append(bins)


    # Plotting results
    plot_bins(
        all_bins,
        times,
        algorithms
        )
    
def test_taboo_search():
    bins = first_fit(items,bin_capacity)

    t0 = time.time()
    solution = taboo_search(bins, 2000)
    t = time.time() - t0
    plot_2_bins([bins, solution], [0,t], ["First Fit", "taboo Search"])



if __name__ == "__main__":

    # command line arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb-items', type=int, default=14, help='number of items')
    parser.add_argument('--max-size', type=int, default=6, help='maximum size')
    parser.add_argument('--bin-capacity', type=int, default=10, help='bin capacity')
    args = parser.parse_args()

    nb_items = args.nb_items
    max_size = args.max_size
    bin_capacity = args.bin_capacity
    
    # print(f"Generating {nb_items} items with max size = {max_size}, bin capacity = {bin_capacity} .")
    # items = genereate_items(nb_items, max_size)
    

    dataset = open("./BPP_50_50_0.1_0.7_0.txt", "r").read().strip().split("\n")
    nb_items = int(dataset[0])
    bin_capacity = int(dataset[1])
    items = []
    for i in range(2, nb_items + 2):
        items.append(Item(int(dataset[i])))
    print(f"Generating {nb_items} items with max size = {max_size}, bin capacity = {bin_capacity} .")

    # test_heuristics()
    test_taboo_search()


    






            