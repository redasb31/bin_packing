import random
import matplotlib as plt
import time
import itertools
from lib.classes import Bin, Item
from lib.utils import plot_bins, genereate_items
from lib.algorithms import first_fit, recursive_branch_and_bound, stack_branch_and_bound, dynamic_branch_and_bound, last_fit, best_fit, worst_fit, next_fit
import argparse

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
    nb_items = 50
    bin_capacity = int(dataset[1])
    items = []
    for i in range(2, nb_items + 2):
        items.append(Item(int(dataset[i])))
    print(f"Generating {nb_items} items with max size = {max_size}, bin capacity = {bin_capacity} .")


    algorithms = ["first_fit", "last_fit", "best_fit", "worst_fit", "next_fit"]
    all_bins = []
    times = []

    for algo in algorithms:
        t0 = time.time()
        bins = eval(algo)(items, bin_capacity)
        t1 = time.time()
        times.append(round(t1 - t0, 6))
        all_bins.append(bins)
        # print(f"{algo} took {times[-1]} seconds")
        # print(f"{algo} used {len(bins)} bins")


    # Plotting results
    plot_bins(
        all_bins,
        times,
        algorithms
        )





            