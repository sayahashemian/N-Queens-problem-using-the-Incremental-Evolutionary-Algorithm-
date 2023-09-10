import random
import time

class NQueenGeneticAlgorithm:
    def __init__(self, population_size, board_size, mutation_rate, tournament_size):
        self.population_size = population_size
        self.board_size = board_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def create_chromosome(self):
        chromosome = list(range(self.board_size))
        random.shuffle(chromosome)
        return chromosome

    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return population

    def fitness(self, chromosome):
        clashes = 0
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == j - i:
                    clashes += 1
        return 1 / (clashes + 1)

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.board_size - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            index1 = random.randint(0, self.board_size - 1)
            index2 = random.randint(0, self.board_size - 1)
            chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]

    def tournament_selection(self, population):
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=self.fitness)
        return winner

    def local_search(self, chromosome):
        best_fitness = self.fitness(chromosome)
        best_chromosome = chromosome
        for i in range(self.board_size):
            for j in range(self.board_size):
                if chromosome[i] != j:
                    new_chromosome = chromosome[:]
                    new_chromosome[i] = j
                    new_fitness = self.fitness(new_chromosome)
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_chromosome = new_chromosome
        return best_chromosome

    def evolve_population(self, population):
        new_population = []
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        return new_population

    def solve(self):
        population = self.create_initial_population()
        generation = 0
        while True:
            best_chromosome = max(population, key=self.fitness)
            if self.fitness(best_chromosome) == 1:
                return best_chromosome
            population = self.evolve_population(population)
            population = [self.local_search(chromosome) for chromosome in population]
            generation += 1

    def print_board(self, chromosome):
        for row in range(self.board_size):
            line = ""
            for column in range(self.board_size):
                if chromosome[column] == row:
                    line += "Q "
                else:
                    line += "- "
            print(line)
        print()

if __name__ == "__main__":
    population_size = 100
    board_size = 8
    mutation_rate = 0.3
    tournament_size = 5

    algorithm = NQueenGeneticAlgorithm(population_size, board_size, mutation_rate, tournament_size)

    start_time = time.time()
    solution = algorithm.solve()
    end_time = time.time()
    execution_time = end_time - start_time

    if solution:
        print(f"Solution found for {board_size}-Queen:")
        algorithm.print_board(solution)
        print("Solution fitness:", algorithm.fitness(solution))
    else:
        print("Solution not found.")

    print("Execution time:", execution_time, "seconds")
