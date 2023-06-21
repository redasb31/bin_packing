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

def first_fit(items,initial_capacity,bins=[]):
    # items = sorted(items)
    # items=items[::-1]
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

def worst_fit(items,initial_capacity,bins=[]):
    # items=items[::-1]
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

def last_fit(items,initial_capacity,bins=[]):
    items=items[::-1]
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

def next_fit(items,initial_capacity,bins=[]):
    items=items[::-1]
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

def best_fit(items,initial_capacity,bins=[]):
    items=items[::-1]
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


        # Create subspaces
        subspaces = create_subspaces(population, num_subspaces)

        # Perform crossover and mutation within each subspace
        new_population = []
        for subspace in subspaces:
            # new_subspace = crossover(subspace, crossover_rate)
            new_subspace = mutation(subspace, mutation_rate)
            new_population += new_subspace

        # Evaluate fitness of the new population
        fitness_scores = [fitness(bins) for bins in new_population]
        # print(fitness_scores);input()
        # Select parents for the next generation
        population = elitist_selection(new_population, fitness_scores, population_size)


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
        if subspace[i]:
            if random.random() < crossover_rate:
                mate = random.choice(subspace)
                while not mate:
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
        if individual:
            for bin in individual:
                if len(bin.items) == 0:
                    individual.remove(bin)
            # Choose a random individual to mutate
            if random.random() < mutation_rate:

                # Choose 2 random bins to mutate
                bin1 = random.choice(individual)
                bin2 = random.choice(individual)
                
                for _ in range(5):
                    if bin1 == bin2:
                        bin2 = random.choice(individual)
                    else:
                        break

                sorted_bins = sorted(individual, key=lambda x: x.capacity)
                chosen_bins_length = random.randint(1, len(sorted_bins) - 1)
                chosen_bins = sorted_bins[:chosen_bins_length]
                ignored_bins = sorted_bins[chosen_bins_length:]
                
                # all the items of the ignored bins
                ignored_items = []
                for bin in ignored_bins:
                    for item in bin.items:
                        ignored_items.append(item)

                individual=best_fit(ignored_items,individual[0].initial_capacity,bins=chosen_bins)
                

                new_subspace.append(individual)
            else:
                new_subspace.append(individual)
    return new_subspace

def elitist_selection(population, fitness_scores, new_len):
    """
    Implements elitist selection for the genetic algorithm.

    Args:
        population (list): List of individuals (bins).
        fitness_scores (list): List of fitness scores for each individual.
        new_len (int): Length of the new population.

    Returns:
        list: List of individuals (bins) selected for the next generation.
    """
    # sort population by fitness_scores
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]

    return sorted_population[:new_len]

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



def grasp(subspace, alpha):
    """
    Perform the GRASP procedure within a subspace.

    Args:
        subspace (list): List of candidate solutions within the subspace.
        alpha (float): Alpha parameter controlling the randomness of GRASP.

    Returns:
        list: A new solution generated by the GRASP procedure.
    """
    candidate_solutions = []
    for solution in subspace:
        candidate_solutions.append(solution)

    best_solution = None
    best_fitness = float('inf')

    while candidate_solutions:
        # Select a candidate solution based on the greedy randomized rule
        candidate = randomized_greedy(candidate_solutions, alpha)
        candidate_fitness = fitness(candidate)

        # Update the best solution if the candidate is better
        if candidate_fitness < best_fitness:
            best_solution = candidate
            best_fitness = candidate_fitness

        # Remove the candidate from the list
        candidate_solutions.remove(candidate)

    return best_solution

def randomized_greedy(candidate_solutions, alpha):
    """
    Select a candidate solution using the randomized greedy rule.

    Args:
        candidate_solutions (list): List of candidate solutions.
        alpha (float): Alpha parameter controlling the randomness.

    Returns:
        list: The selected candidate solution.
    """
    candidate_scores = [fitness(candidate) for candidate in candidate_solutions]

    # Calculate the threshold for candidate selection
    min_score = min(candidate_scores)
    max_score = max(candidate_scores)
    threshold = min_score + alpha * (max_score - min_score)

    # Filter candidates with scores below the threshold
    eligible_candidates = [
        candidate for candidate, score in zip(candidate_solutions, candidate_scores) if score <= threshold
    ]

    # Select a random candidate from the eligible candidates
    return random.choice(eligible_candidates)


def hybrid_algorithm(
        items, bin_capacity, num_subspaces, population_size, num_generations,
        crossover_rate, mutation_rate, grasp_iterations, grasp_alpha):
    """
    Perform bin packing with a hybrid algorithm combining dispersed genetic algorithm and GRASP.

    Args:
        items (list): List of Item objects to be packed.
        bin_capacity (int): Capacity of each bin.
        num_subspaces (int): Number of subspaces to create.
        population_size (int): Size of the population.
        num_generations (int): Number of generations to run the algorithm for.
        crossover_rate (float): Rate at which to perform crossover.
        mutation_rate (float): Rate at which to perform mutation.
        grasp_iterations (int): Number of GRASP iterations per population.
        grasp_alpha (float): Alpha parameter for GRASP (controls the randomness).

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

        # Perform GRASP within each subspace
        new_population = []
        for subspace in subspaces:
            for _ in range(grasp_iterations):
                grasp_solution = grasp(subspace, grasp_alpha)
                new_population.append(grasp_solution)
                

        # Perform crossover and mutation within the new population
        if new_population:
            #new_population = crossover(new_population, crossover_rate)
            new_population = mutation(new_population, mutation_rate)


        # Evaluate fitness of the new population
        fitness_scores = [fitness(bins) for bins in new_population]

        # Select parents for the next generation
        population = elitist_selection(new_population, fitness_scores, population_size)

    # Evaluate the final population and return the best individual
    fitness_scores = [fitness(bins) for bins in population]
    best_index, best_fitness = max(enumerate(fitness_scores), key=lambda x: x[1])
    best_individual = population[best_index]

    return best_fitness, best_individual
