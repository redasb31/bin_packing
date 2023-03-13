import random
import matplotlib as plt
import time
import itertools
from lib.classes import Bin, Item
from lib.utils import plot_bins, genereate_items
from lib.algorithms import first_fit, recursive_branch_and_bound, stack_branch_and_bound, dynamic_branch_and_bound
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
    
    print(f"Generating {nb_items} items with max size = {max_size}, bin capacity = {bin_capacity} .")
    items = genereate_items(nb_items, max_size)

    # first fit
    t0 = time.time()
    bins = first_fit(items, bin_capacity)
    t0 = round(time.time()-t0, 6)

    # branch and bound with stack
    t1 = time.time()
    bins1 = stack_branch_and_bound(items, bins, bin_capacity)
    t1 = round(time.time() - t1, 6)

    # recursive branch and bound
    t2 = time.time()
    bins2 = recursive_branch_and_bound(items, [], bins)
    t2 = round(time.time() - t2, 6)

    # branch and bound with stack using dynamic programming
    t3 = time.time()
    bins3 = dynamic_branch_and_bound(items, bins, bin_capacity)
    t3 = round(time.time()- t3, 6)

    # Plotting results
    plot_bins(
        [bins, bins1, bins2, bins3],
        [t0, t1, t2, t3],
        ['First Fit', 'BNB avec Pile', 'BNB Récursive', 'BNB Dynamique']
        )




            