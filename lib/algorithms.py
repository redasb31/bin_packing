from lib.utils import state_hash, plot_1_bin, plot_2_bins
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
    # items=items[::-1]
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

def taboo_search(bins, nb_iterations, taboo_list_size):
    best_solution = bins
    tabu_list = []
    tabu_list.append(bins)
    for i in range(nb_iterations):
        new_bins = bins.copy()
        # on vérifie que le voisin n'est pas dans la liste tabou
        new_bins = generate_neighbor(new_bins)
        if new_bins in tabu_list:
                continue
        # si on a pas trouvé de voisin non tabou, on prend le premier voisin

        if evaluate(new_bins) < evaluate(bins):
            bins = new_bins
            if evaluate(bins) < evaluate(best_solution):
                best_solution = bins
        else:
            # random value between 0 and 1

            if random.random() < math.exp((evaluate(bins) - evaluate(new_bins)) / nb_iterations):
                bins = new_bins
        
        tabu_list=tabu_list[:taboo_list_size]
        tabu_list.append(bins)
        
    return best_solution

def dispersed_genetic_algorithm(
        items, bin_capacity, num_subspaces, population_size, num_generations,
        crossover_rate, mutation_rate):
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
    population = initialize_population(items, bin_capacity, population_size)

    for generation in range(num_generations):
        # Evaluate fitness of the population
        fitness_scores = [fitness(bins) for bins in population]
        # Find the best individual in the population
        best_index, best_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
        best_individual = population[best_index]

        # Check if we have met the stop criterion
        if best_fitness == 0:
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
        fitness_scores = [fitness(bins) for bins in new_population]
        # print(fitness_scores);input()
        # Select parents for the next generation
        population = [tournament_selection(new_population, fitness_scores) for _ in range(population_size)]


    # Evaluate the final population and return the best individual
    fitness_scores = [fitness(bins) for bins in population]
    best_index, best_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
    best_individual = population[best_index]

    return best_fitness, best_individual

def initialize_population(items, bin_capacity, population_size):
    """
    Initializes a population of individuals for the dispersed genetic algorithm.

    Args:
        items (list): List of Item objects to be packed.
        bin_capacity (int): Capacity of each bin.
        population_size (int): Size of the population.

    Returns:
        list: List of individuals (bins).
    """
    population = []
    for individual in range(population_size):
        bins = []
        for _ in range(100):
            bin_obj = Bin(bin_capacity)
            bins.append(bin_obj)
        
        # Randomly assign items to bins
        random_bin_assignment(items, bins)
        
        # plot_1_bin(bins, 0, f"Individual {individual}")
        population.append(bins)

    
    return population

def random_bin_assignment(items, bins):
    """
    Assigns the given items to bins randomly.

    items: list of Item objects to be packed into bins
    num_bins: number of bins available

    Returns a list of Bin objects representing the bins with packed items.
    """
    random.shuffle(items)
    for item in items:
        for bin in bins:
            if bin.capacity >= item.size:
                bin.add_item(item)
                break
    return bins

def fitness(bins):
    """
    Computes the fitness of a bin assignment, which is the sum of the capacities of the bins used.

    bins: list of Bin objects representing the bins with packed items

    Returns the fitness value.
    """
    return sum([bin.initial_capacity - bin.capacity for bin in bins])

def crossover(subspace, crossover_rate):
    """
    Implements a crossover operation for the bin packing problem.

    subspace: list of Individuals (bins) in the subspace
    crossover_rate: rate at which to perform crossover

    Returns a new subspace with crossover applied.
    """
    new_subspace = []
    for i in range(len(subspace)):
        # Choose a random individual to mate with
        if random.random() < crossover_rate:
            mate = random.choice(subspace)
            new_individual = []
            for j in range(len(subspace[i])):
                # if a bin is empty, delete it
                if len(subspace[i][j].items) == 0:
                    continue

                # Choose a random bin to inherit from the mate
                if random.random() < 0.5:
                    new_individual.append(subspace[i][j])
                else:
                    new_individual.append(mate[j])
            new_subspace.append(new_individual)
            # plot_2_bins([mate, new_individual], [0,0], ["the chosen individual", "the new individual"])

        else:
            new_subspace.append(subspace[i])
        
    return new_subspace
    
def mutation(subspace, mutation_rate):
    """
    Implements a mutation operation for the bin packing problem.

    subspace: list of Individuals (bins) in the subspace
    mutation_rate: rate at which to perform mutation

    Returns a new subspace with mutation applied.
    """
    new_subspace = []
    for individual in subspace:
        # Choose a random individual to mutate
        if random.random() < mutation_rate:
            for _ in range(10):
                # Choose 2 random bins to mutate
                bin1 = random.choice(individual)
                bin2 = random.choice(individual)

                # mutate 2 random bins by inverting 2 random items from each bin
                bin1_items = bin1.items
                bin2_items = bin2.items
                if len(bin1_items) > 0 and len(bin2_items) > 0:
                    item1 = random.choice(bin1_items)
                    item2 = random.choice(bin2_items)
                    if item1.size + bin2.capacity - item2.size <= bin2.initial_capacity and item2.size + bin1.capacity - item1.size <= bin1.initial_capacity:
                        bin1.remove_item(item1)
                        bin1.add_item(item2)
                        bin2.remove_item(item2)
                        bin2.add_item(item1)
                        break

            new_subspace.append(individual)
        else:
            new_subspace.append(individual)
    return new_subspace

def tournament_selection(population, fitness_scores):
    """
    Implements tournament selection for the genetic algorithm.

    Args:
        population (list): List of individuals (bins).
        fitness_scores (list): List of fitness scores for each individual.

    Returns:
        list: List of individuals (bins) selected for the next generation.
    """
    # Return the individual with the highest fitness score
    return max(population, key=lambda x: fitness_scores[population.index(x)])

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


def hybrid_algorithm(
        items, bin_capacity, num_subspaces, population_size, num_generations,
        crossover_rate, mutation_rate, dragonfly_iterations):
    """
    Perform a hybrid algorithm by combining Dispersed Genetic Algorithm with Dragonfly Algorithm.

    Args:
        items (list): List of Item objects to be packed.
        bin_capacity (int): Capacity of each bin.
        num_subspaces (int): Number of subspaces to create.
        population_size (int): Size of the population.
        num_generations (int): Number of generations to run the algorithm for.
        crossover_rate (float): Rate at which to perform crossover in the genetic algorithm phase.
        mutation_rate (float): Rate at which to perform mutation in the genetic algorithm phase.
        dragonfly_iterations (int): Number of iterations to run the dragonfly algorithm phase.

    Returns:
        tuple: A tuple containing the best fitness score and the corresponding list of bins.
    """
    population = initialize_population(items, bin_capacity, population_size)

    for generation in range(num_generations):
        # Evaluate fitness of the population
        fitness_scores = [fitness(bins) for bins in population]
        # Find the best individual in the population
        best_index, best_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
        best_individual = population[best_index]

        # Check if we have met the stop criterion
        if best_fitness == 0:
            return best_fitness, best_individual

        # Genetic Algorithm phase
        subspaces = create_subspaces(population, num_subspaces)

        new_population = []
        for subspace in subspaces:
            new_subspace = crossover(subspace, crossover_rate)
            new_subspace = mutation(new_subspace, mutation_rate)
            new_population += new_subspace

        population = [tournament_selection(new_population, fitness_scores) for _ in range(population_size)]

        # Dragonfly Algorithm phase
        dragonflies = population.copy()

        for _ in range(dragonfly_iterations):
            for i in range(population_size):
                # Update dragonfly's position using Dragonfly Algorithm rules
                update_position(population, dragonflies[i])

                # Evaluate fitness of the updated position
                new_fitness = fitness(dragonflies[i])
                # If the new position improves fitness, update it
                if new_fitness < fitness_scores[i]:
                    population[i] = dragonflies[i]
                    fitness_scores[i] = new_fitness

    # Evaluate the final population and return the best individual
    fitness_scores = [fitness(bins) for bins in population]
    best_index, best_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
    best_individual = population[best_index]

    return best_fitness, best_individual


def update_position(dragonfly, population, calculate_dissimilarity):
    """
    Update the position of a dragonfly based on the Dragonfly Algorithm rules.

    Args:
        dragonfly (list): List representing the position of a dragonfly.
        population (list): List of all dragonflies in the population.
        calculate_dissimilarity (function): Function to calculate the dissimilarity between two bin packing solutions.
    """
    step_size = 0.1  # Step size for random movement
    attractiveness_weight = 1.0  # Weight for attractiveness
    dissimilarity_weight = 1.0  # Weight for dissimilarity

    random_dragonfly = random.choice(population)

    attractiveness = attractiveness_weight / (1.0 + calculate_dissimilarity(dragonfly, random_dragonfly))

    dissimilarity = calculate_dissimilarity(dragonfly, random_dragonfly)

    # Update the position
    dragonfly[:] = [x + step_size * (attractiveness * (y - x) + dissimilarity_weight * dissimilarity * (random.random() - 0.5))
                    for x, y in zip(dragonfly, random_dragonfly)]

def calculate_dissimilarity(solution1, solution2):
    """
    Calculate the dissimilarity between two bin packing solutions based on their total utilization.

    Args:
        solution1 (list): First bin packing solution (list of bins).
        solution2 (list): Second bin packing solution (list of bins).

    Returns:
        float: Dissimilarity between the two solutions.
    """
    total_utilization1 = sum(bin_utilization(bin) for bin in solution1)
    total_utilization2 = sum(bin_utilization(bin) for bin in solution2)

    dissimilarity = abs(total_utilization1 - total_utilization2)

    return dissimilarity

def bin_utilization(bin):
    """
    Calculate the utilization of a bin.

    Args:
        bin (list): List representing a bin.

    Returns:
        float: Utilization of the bin.
    """

    return bin.initial_capacity - bin.capacity

# Greedy Randomized Adaptive Search Procedure
def GRASP(items, bin_capacity, alpha, max_iterations):
    """
    Perform bin packing with a GRASP algorithm.

    Args:
        items (list): List of Item objects to be packed.
        bin_capacity (int): Capacity of each bin.
        alpha (float): Alpha value for GRASP.
        max_iterations (int): Maximum number of iterations to run the algorithm for.

    Returns:
        tuple: A tuple containing the best fitness score and the corresponding list of bins.
    """
    best_fitness = float('inf')
    best_solution = None

    for _ in range(max_iterations):
        # Construct a greedy solution
        solution = greedy_randomized_construction(items, bin_capacity, alpha)

        # Perform local search
        solution = local_search(solution, bin_capacity)

        # Update the best solution
        fitness_score = fitness(solution)
        if fitness_score < best_fitness:
            best_fitness = fitness_score
            best_solution = solution

    return best_fitness, best_solution

# greedy_randomized_construction
def greedy_randomized_construction(items, bin_capacity, alpha):
    """
    Construct a greedy solution using GRASP.

    Args:
        items (list): List of Item objects to be packed.
        bin_capacity (int): Capacity of each bin.
        alpha (float): Alpha value for GRASP.

    Returns:
        list: List of bins representing the greedy solution.
    """
    # Initialize a list of bins
    bins = []
    for _ in range(100):
        bin_obj = Bin(bin_capacity)
        bins.append(bin_obj)

    # Initialize a list of items
    items = items.copy()

    # Construct the greedy solution
    while len(items) > 0:
        # Construct a restricted candidate list (RCL) of items
        rcl = construct_rcl(items, bins, alpha)

        # Choose an item from the RCL at random
        item = random.choice(rcl)

        # Add the item to the bin with the most capacity
        max_capacity = max(bin.capacity for bin in bins)
        for bin in bins:
            if bin.capacity == max_capacity:
                bin.add_item(item)
                break

        # Remove the item from the list of items
        items.remove(item)

    return bins

# construct_rcl
def construct_rcl(items, bins, alpha):
    """
    Construct a restricted candidate list (RCL) of items.

    Args:
        items (list): List of Item objects to be packed.
        bins (list): List of bins.
        alpha (float): Alpha value for GRASP.

    Returns:
        list: List of items in the RCL.
    """
    # Calculate the utilization of each bin
    utilizations = [bin_utilization(bin) for bin in bins]

    # Calculate the average utilization
    average_utilization = sum(utilizations) / len(utilizations)

    # Calculate the threshold utilization
    threshold_utilization = average_utilization + alpha * (max(utilizations) - average_utilization)

    # Construct the RCL
    rcl = [item for item in items if item.size <= threshold_utilization]

    return rcl

# local_search
def local_search(solution, bin_capacity):
    """
    Perform local search on a solution.

    Args:
        solution (list): List of bins representing the solution.
        bin_capacity (int): Capacity of each bin.

    Returns:
        list: List of bins representing the solution after local search.
    """
    # Initialize a list of bins
    bins = []
    for _ in range(100):
        bin_obj = Bin(bin_capacity)
        bins.append(bin_obj)

    # Initialize a list of items
    items = []
    for bin in solution:
        for item in bin.items:
            items.append(item)

    # Construct the greedy solution
    while len(items) > 0:
        # Construct a restricted candidate list (RCL) of items
        rcl = construct_rcl(items, bins, 0)

        # Choose an item from the RCL at random
        item = random.choice(rcl)

        # Add the item to the bin with the most capacity
        max_capacity = max(bin.capacity for bin in bins)
        for bin in bins:
            if bin.capacity == max_capacity:
                bin.add_item(item)
                break

        # Remove the item from the list of items
        items.remove(item)

    return bins