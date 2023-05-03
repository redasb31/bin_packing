from lib.utils import state_hash
from lib.classes import Bin, Item
import random, math

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
        if (state_hash(items, bins)) in visited:
            continue
        visited.append(state_hash((items),bins) )
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
    # items = sorted(items)
    items=items[::-1]
    bins = []
    for item in items:
        for bin in bins:
            if bin.capacity >= item.size:
                bin.add_item(item)
                break
        else:
            bin = Bin(initial_capacity)
            bin.add_item(item)
            bins.append(bin)
    return bins


def worst_fit(items,initial_capacity):
    # items=items[::-1]
    bins = []
    for item in items:
        worst_bin = None
        for bin in bins:
            if bin.capacity >= item.size:
                if worst_bin is None or bin.capacity > worst_bin.capacity:
                    worst_bin = bin
        if worst_bin is not None:
            worst_bin.add_item(item)
        else:
            bin = Bin(initial_capacity)
            bin.add_item(item)
            bins.append(bin)
    return bins


def last_fit(items,initial_capacity):
    items=items[::-1]
    bins = []
    for item in items:
        last_bin = None
        for bin in bins:
            if bin.capacity >= item.size:
                last_bin = bin
        if last_bin is not None:
            last_bin.add_item(item)
        else:
            bin = Bin(initial_capacity)
            bin.add_item(item)
            bins.append(bin)
    return bins


def next_fit(items,initial_capacity):
    items=items[::-1]
    bins = []
    bin = Bin(initial_capacity)
    for item in items:
        if bin.capacity >= item.size:
            bin.add_item(item)
        else:
            bins.append(bin)
            bin = Bin(initial_capacity)
            bin.add_item(item)
    bins.append(bin)
    return bins


def best_fit(items,initial_capacity):
    items=items[::-1]
    bins = []
    for item in items:
        best_bin = None
        for bin in bins:
            if bin.capacity >= item.size:
                if best_bin is None or bin.capacity < best_bin.capacity:
                    best_bin = bin
        if best_bin is not None:
            best_bin.add_item(item)
        else:
            bin = Bin(initial_capacity)
            bin.add_item(item)
            bins.append(bin)
    return bins


def generate_neighbor(solution):
    n_bins = len(solution)
    bin_idx = random.randint(0,n_bins-1)  # choisir un conteneur au hasard
    if len(solution[bin_idx]) == 0:
        solution.pop(bin_idx)
        return solution  # pas de voisin possible si le conteneur est vide
    
    item_idx = random.randint(0,len(solution[bin_idx])-1)  # choisir un objet au hasard dans le conteneur
    new_bin_idx = random.randint(0,n_bins-1)  # choisir un nouveau conteneur au hasard
    while new_bin_idx == bin_idx:  # éviter de choisir le même conteneur
        # print(new_bin_idx, bin_idx)
        new_bin_idx = random.randint(0,n_bins-1)

    neighbor = solution.copy()
    item = neighbor[bin_idx].items[item_idx]
    if neighbor[new_bin_idx].capacity >= item.size:
        neighbor[bin_idx].remove_item(item)  
        neighbor[new_bin_idx].add_item(item)  
    
    return neighbor


#metaheuristic taboo search
def taboo_search(bins, nb_iterations):
    best_solution = bins
    tabu_list = []
    tabu_list.append(bins)
    for i in range(nb_iterations):
        new_bins = bins.copy()
        # on vérifie que le voisin n'est pas dans la liste tabou
        for _ in range(10):
            new_bins = generate_neighbor(new_bins)
            if new_bins not in tabu_list:
                break
        # si on a pas trouvé de voisin non tabou, on prend le premier voisin

        if evaluate(new_bins) < evaluate(bins):
            bins = new_bins
            if evaluate(bins) < evaluate(best_solution):
                best_solution = bins
        else:
            # random value between 0 and 1

            if random.random() < math.exp((evaluate(bins) - evaluate(new_bins)) / nb_iterations):
                bins = new_bins
        
        tabu_list.append(bins)
        
    return best_solution

