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

def dispersed_genetic_algorithm(
        items, bin_capacity, num_subspaces, population_size, num_generations,
        crossover_rate, mutation_rate, selection_function, fitness_function,
        stop_criterion):
    """
    Perform bin packing with a dispersed genetic algorithm.

    Args:
        items (list): List of Item objects to be packed.
        bin_capacity (int): Capacity of each bin.
        num_subspaces (int): Number of subspaces to create.
        population_size (int): Size of the population.
        num_generations (int): Number of generations to run the algorithm for.
        crossover_rate (float): Rate at which to perform crossover.
        mutation_rate (float): Rate at which to perform mutation.
        selection_function (function): Selection function for selecting parents for mating.
        fitness_function (function): Fitness function for evaluating the fitness of a population.
        stop_criterion (function): Function that takes in the current generation and the current best fitness and returns
            a Boolean value indicating whether the algorithm should stop.

    Returns:
        tuple: A tuple containing the best fitness score and the corresponding list of bins.
    """
    population = initialize_population(items, bin_capacity, num_subspaces, population_size)

    for generation in range(num_generations):
        # Evaluate fitness of the population
        fitness_scores = [fitness_function(bins, bin_capacity) for bins in population]

        # Find the best individual in the population
        best_index, best_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
        best_individual = population[best_index]

        # Check if we have met the stop criterion
        if stop_criterion(generation, best_fitness):
            return best_fitness, best_individual

        # Create subspaces
        subspaces = create_subspaces(population, num_subspaces)

        # Perform crossover and mutation within each subspace
        new_population = []
        for subspace in subspaces:
            new_subspace = crossover(subspace, crossover_rate)
            new_subspace = mutation(new_subspace, mutation_rate)
            new_population += new_subspace

        # Evaluate fitness of the new population
        fitness_scores = [fitness_function(bins, bin_capacity) for bins in new_population]

        # Select parents for the next generation
        parents = [selection_function(new_population, fitness_scores) for _ in range(population_size)]

        # Create the next generation by recombination
        population = recombine(parents)

    # Evaluate the final population and return the best individual
    fitness_scores = [fitness_function(bins, bin_capacity) for bins in population]
    best_index, best_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
    best_individual = population[best_index]

    return best_fitness, best_individual

def random_bin_assignment(items, num_bins):
    """
    Assigns the given items to bins randomly.

    items: list of Item objects to be packed into bins
    num_bins: number of bins available

    Returns a list of Bin objects representing the bins with packed items.
    """
    bins = [Bin(capacity=1) for i in range(num_bins)]
    random.shuffle(items)
    for item in items:
        for bin in bins:
            if bin.capacity >= item.size:
                bin.add_item(item)
                break
        else:
            raise ValueError("No bin has enough capacity to hold item")
    return bins

def fitness(bins):
    """
    Computes the fitness of a bin assignment, which is the sum of the capacities of the bins used.

    bins: list of Bin objects representing the bins with packed items

    Returns the fitness value.
    """
    return sum([bin.initial_capacity - bin.capacity for bin in bins])

def crossover(parent1, parent2):
    """
    Implements a crossover operation for the bin packing problem.

    parent1: list of Bin objects representing the bins assigned to items in the first parent
    parent2: list of Bin objects representing the bins assigned to items in the second parent

    Returns two new offspring created by swapping bins between the parents.
    """
    # Choose a random bin to swap between the parents
    bin_index = random.randint(0, len(parent1)-1)

    # Create the first offspring by swapping the bin at the chosen index
    offspring1 = parent1[:bin_index] + [parent2[bin_index]] + parent1[bin_index+1:]

    # Create the second offspring by swapping the bin at the chosen index
    offspring2 = parent2[:bin_index] + [parent1[bin_index]] + parent2[bin_index+1:]

    return offspring1, offspring2

def mutation(individual, items):
    """
    Implements a mutation operation for the bin packing problem.

    individual: list of Bin objects representing the bins assigned to items in the individual
    items: list of Item objects to be packed into bins

    Returns a new individual with a randomly chosen item reassigned to a different bin.
    """
    # Choose a random item to reassign
    item_index = random.randint(0, len(items)-1)
    item = items[item_index]

    # Choose a random bin to reassign the item to
    bin_index = random.randint(0, len(individual)-1)
    new_bin = individual[bin_index]

    # Check if the item fits in the new bin, and if not, try the other bins in random order
    if item.size > new_bin.capacity:
        candidate_bins = individual[:bin_index] + individual[bin_index+1:]
        random.shuffle(candidate_bins)
        for candidate_bin in candidate_bins:
            if item.size <= candidate_bin.capacity:
                new_bin = candidate_bin
                break

    # Remove the item from its current bin and add it to the new bin
    for bin in individual:
        if item in bin.items:
            bin.remove_item(item)
            break
    new_bin.add_item(item)

    return individual

def recombine(parent1, parent2):
    """Performs crossover between two parents to generate two children"""
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def tournament_selection(population, tournament_size):
    """
    Implements a tournament selection operation for the bin packing problem.

    population: list of individuals to select from
    tournament_size: size of the tournament to be held

    Returns the winning individual selected by the tournament.
    """
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=fitness)


def initialize_population(items, bin_capacity, num_subspaces, population_size):
    """
    Initializes a population of individuals for the dispersed genetic algorithm.

    Args:
        items (list): List of Item objects to be packed.
        bin_capacity (int): Capacity of each bin.
        num_subspaces (int): Number of subspaces to create.
        population_size (int): Size of the population.

    Returns:
        list: List of individuals (bins).
    """
    population = []
    for _ in range(population_size):
        bins = []
        for _ in range(num_subspaces):
            bin_obj = Bin(bin_capacity)
            bins.append(bin_obj)
        
        # Randomly assign items to bins
        random_bin_assignment(items, bins)
        
        population.append(bins)
    
    return population


def create_subspaces(population, num_subspaces):
    """
    Divides the population into subspaces for the dispersed genetic algorithm.

    Args:
        population (list): List of individuals (bins).
        num_subspaces (int): Number of subspaces to create.

    Returns:
        list: List of subspaces, each containing individuals (bins).
    """
    subspaces = [[] for _ in range(num_subspaces)]

    for bins in population:
        subspace_index = random.randint(0, num_subspaces - 1)
        subspaces[subspace_index].append(bins)

    return subspaces