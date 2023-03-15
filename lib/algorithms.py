from lib.utils import state_hash
from lib.classes import Bin, Item


def evaluate(bins):
    return len(bins)

def dynamic_branch_and_bound(items, best_solution,initial_capacity):
    items.sort()
    stack = [(items,[])]
    visited=[]
    while(len(stack)>0):

        NA=stack.pop()
        items,bins=NA

        if evaluate(bins)>=evaluate(best_solution):
            continue

        if (state_hash(items, bins)) in visited:
            continue
        visited.append(state_hash((items),bins) )

        if (len(items)==0):
            if (evaluate(bins)<evaluate(best_solution)) :
                best_solution=bins
            continue

        item=items[0]

        new_bins = bins.copy()
        new_bin = Bin(initial_capacity)
        new_bin.add_item(item)
        new_bins.append(new_bin)
        new_items=items.copy()
        new_items.remove(item)
        stack.append((new_items,new_bins))

        for bin in bins:
                if bin.capacity>=item.size:
                    new_bins = bins.copy()
                    new_bins.remove(bin)
                    new_bin = Bin(bin.initial_capacity)
                    new_bin.capacity=bin.capacity
                    new_bin.items = bin.items.copy()
                    new_bin.add_item(item)
                    new_bins.append(new_bin)
                    new_items=items.copy()
                    new_items.remove(item)
                    stack.append((new_items,new_bins))
            
    return best_solution


def stack_branch_and_bound(items, best_solution,initial_capacity):
    stack = [(items,[])]
    while(len(stack)>0):

        NA=stack.pop()
        items,bins=NA

        if evaluate(bins)>=evaluate(best_solution):
            continue

        if (len(items)==0):
            if (evaluate(bins)<evaluate(best_solution)) :
                best_solution=bins
            continue

        item=items[0]

        new_bins = bins.copy()
        new_bin = Bin(initial_capacity)
        new_bin.add_item(item)
        new_bins.append(new_bin)
        new_items=items.copy()
        new_items.remove(item)
        stack.append((new_items,new_bins))

        for bin in bins:
                if bin.capacity>=item.size:
                    new_bins = bins.copy()
                    new_bins.remove(bin)
                    new_bin = Bin(bin.initial_capacity)
                    new_bin.capacity=bin.capacity
                    new_bin.items = bin.items.copy()
                    new_bin.add_item(item)
                    new_bins.append(new_bin)
                    new_items=items.copy()
                    new_items.remove(item)
                    stack.append((new_items,new_bins))
    return best_solution

def recursive_branch_and_bound(items, bins, best_solution):
    if len(items) == 0:
        if len(bins) < len(best_solution):
            best_solution = bins
            # print(f"New best solution: {len(bins)} bins")
        return best_solution

    # try to add item to each bin
    for bin in bins:
        if bin.capacity >= items[0].size:
            new_bins = bins.copy()
            new_bins.remove(bin)
            new_bin = Bin(bin.initial_capacity)
            new_bin.capacity=bin.capacity
            new_bin.items = bin.items.copy()
            new_bin.add_item(items[0])
            new_bins.append(new_bin)
            best_solution = recursive_branch_and_bound(items[1:], new_bins, best_solution)

    # create new bin
    new_bins = bins.copy()
    new_bin = Bin(10)
    new_bin.add_item(items[0])
    new_bins.append(new_bin)
    best_solution = recursive_branch_and_bound(items[1:], new_bins, best_solution)

    return best_solution

def first_fit(items,initial_capacity):
    new_items=items.copy()
    new_items.sort()
    new_items=new_items[::-1]
    bins = []
    for item in new_items:
        for bin in bins:
            if bin.capacity >= item.size:
                bin.add_item(item)
                break
        else:
            bin = Bin(initial_capacity)
            bin.add_item(item)
            bins.append(bin)
    return bins
