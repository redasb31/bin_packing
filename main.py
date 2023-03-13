import random
import matplotlib as plt
import time
import itertools
from classes import Bin, Item
from utils import plot_bins


def dynamic_branch_and_bound(items, best_solution,initialcapacity):
    items.sort()
    stack = [(items,[])]
    visited=[]
    while(len(stack)>0):

        NA=stack.pop()
        items,bins=NA

        if (state_hash(items, bins)) in visited:
            continue
        visited.append(state_hash((items),bins) )

        if (len(items)==0):
            if (len(bins)<len(best_solution)) :
                best_solution=bins
            continue

        item=items[0]

        if(len(bins))<(len(best_solution)-1):
                new_bins = bins.copy()
                new_bin = Bin(initialcapacity)
                new_bin.add_item(item)
                new_bins.append(new_bin)
                new_items=items.copy()
                new_items.remove(item)
                stack.append((new_items,new_bins))
        for bin in bins:
                if bin.capacity>=item.size:
                    new_bins = bins.copy()
                    new_bins.remove(bin)
                    new_bin = Bin(bin.initialcapacity)
                    new_bin.capacity=bin.capacity
                    new_bin.items = bin.items.copy()
                    new_bin.add_item(item)
                    new_bins.append(new_bin)
                    new_items=items.copy()
                    new_items.remove(item)
                    stack.append((new_items,new_bins))
            
    return best_solution



def stack_branch_and_bound(items, best_solution,initialcapacity):
    stack = [(items,[])]
    while(len(stack)>0):

        NA=stack.pop()
        items,bins=NA

        if (len(items)==0):
            if (len(bins)<len(best_solution)) :
                best_solution=bins
            continue

        item=items[0]

        if(len(bins))<(len(best_solution)-1):
                new_bins = bins.copy()
                new_bin = Bin(initialcapacity)
                new_bin.add_item(item)
                new_bins.append(new_bin)
                new_items=items.copy()
                new_items.remove(item)
                stack.append((new_items,new_bins))
        for bin in bins:
                if bin.capacity>=item.size:
                    new_bins = bins.copy()
                    new_bins.remove(bin)
                    new_bin = Bin(bin.initialcapacity)
                    new_bin.capacity=bin.capacity
                    new_bin.items = bin.items.copy()
                    new_bin.add_item(item)
                    new_bins.append(new_bin)
                    new_items=items.copy()
                    new_items.remove(item)
                    stack.append((new_items,new_bins))
    return best_solution

def state_hash(items,bins):
    set_bins=set(bins)
    s=hex(len(items))[2:]
    for bin in set_bins:
        s=s+hex(bin.capacity)[2:]
    return s

def branch_and_bound(items, bins, best_solution):
    if len(items) == 0:
        if len(bins) < len(best_solution):
            best_solution = bins
            print(f"New best solution: {len(bins)} bins")
        return best_solution

    # try to add item to each bin
    for bin in bins:
        if bin.capacity >= items[0].size:
            new_bins = bins.copy()
            new_bins.remove(bin)
            new_bin = Bin(bin.initialcapacity)
            new_bin.capacity=bin.capacity
            new_bin.items = bin.items.copy()
            new_bin.add_item(items[0])
            new_bins.append(new_bin)
            best_solution = branch_and_bound(items[1:], new_bins, best_solution)

    # create new bin
    new_bins = bins.copy()
    new_bin = Bin(10)
    new_bin.add_item(items[0])
    new_bins.append(new_bin)
    best_solution = branch_and_bound(items[1:], new_bins, best_solution)

    return best_solution


def print_result(items,bins):
    print(f"Number of bins: {len(bins)}")
    print(f"Bins capacity: {bins[0].initialcapacity}")
    print(f"Items: {[str(item) for item in items]}")
    for bin in bins:
        print(bin)
        bin.print_items()

def first_fit(items,initialcapacity):
    new_items=items.copy()
    new_items.sort()
    bins = []
    for item in new_items:
        for bin in bins:
            if bin.capacity >= item.size:
                bin.add_item(item)
                break
        else:
            bin = Bin(initialcapacity)
            bin.add_item(item)
            bins.append(bin)
    return bins

def genereate_items(maxsize,number):
    items = []
    for i in range(number):
        items.append(Item(random.randint(1, maxsize)))
    return items

if __name__ == "__main__":
    # instantiate items with random size
    items=genereate_items(6, 14)
    # bin packing algorithm
    tff=time.time()
    bins = first_fit(items, 10)
    tff=time.time()-tff
    # print results

    t1=time.time()
    bins1=stack_branch_and_bound(items, bins,10)
    t1=-(t1-time.time())
    t2=time.time()
    bins2=branch_and_bound(items, [], bins)
    t2=-(t2-time.time())
    t3=time.time()
    bins3=dynamic_branch_and_bound(items, bins,10)
    t3=-(t3-time.time())


    print(f"Execution time: {t1} seconds")
    print(f"Execution time: {t2} seconds")
    print(f"Execution time: {t3} seconds")
    print(len(bins1))
    print(len(bins3))
    plot_bins((bins,bins1,bins2,bins3),(tff,t1,t2,t3),('First Fit','BNB avec Pile','BNB Récursive','BNB Dynamique'))




            