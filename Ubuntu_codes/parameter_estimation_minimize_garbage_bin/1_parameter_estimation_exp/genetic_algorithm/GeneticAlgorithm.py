"""
Genetic Algorithm code.
"""

import numpy as np
import random
from copy import deepcopy


class GeneticAlgorithm:

    def __init__(self, gene_size, chromosome_size):
        self.geneSize = gene_size
        self.chromosomeSize = chromosome_size
        self.population = np.random.randint(2, size=(self.chromosomeSize, self.geneSize))
        self.new_population = np.zeros([self.chromosomeSize, self.geneSize])
        self.fitness = []
        self.fitnessPercent = []
        self.roulette = np.zeros(chromosome_size)
        self.parent = []
        self.generation = 0

    def __repr__(self):
        return 'GeneticAlgorithm({}, {})'.format(self.geneSize, self.chromosomeSize)

    def __str__(self):
        return 'GA simulation with {} genes, {} chromosomes in the {}th generation'.format(self.geneSize,
                                                                                           self.chromosomeSize,
                                                                                           self.generation)

    def populate(self, selection, limit):
        weight = 0
        weight_violation = 0

        for i in range(self.chromosomeSize):
            for j in range(self.geneSize):
                if self.population[i, j] == 1:
                    weight = weight + selection[0, j]

                if weight > limit:
                    self.reroll_chromosome(i)
                    weight_violation = weight_violation + 1
            weight = 0

        # This can be optimized because it is currently re-evaluating all weights if one set is incorrect.
        if weight_violation > 0:
            self.populate(selection, limit)

    """
    Evaluation of the fitness of each chromosome.
    
    Selection: Table of all items' weight and its corresponding score in the format: (Item weight, Score)
    Limit: Max weight
    """

    def fitness_calc(self, selection):
        fitness_individual = 0

        for i in range(self.chromosomeSize):
            for j in range(self.geneSize):
                if self.population[i, j] == 1:
                    fitness_individual = fitness_individual + selection[1, j]
                else:
                    pass
            self.fitness.append(fitness_individual)
            fitness_individual = 0
        pass

    def reroll_chromosome(self, chromosome_number):
        for i in range(self.geneSize):
            self.population[chromosome_number, i] = np.random.randint(0, 2)

    def roulette_calc(self):
        total_fitness = np.sum(self.fitness)

        for i in range(len(self.fitness)):
            self.fitnessPercent = self.fitness / total_fitness

        try:
            for i in range(len(self.fitness)):
                self.roulette[i] = np.sum(self.fitnessPercent[0:i + 1])

        # The last value
        except IndexError:
            self.roulette[-1] = 1

    def parent_selection(self):
        for i in range(len(self.roulette)):
            number = np.random.rand()

            # Loop around all the roulette %s, finding the highest one without going over
            for j in range(len(self.roulette)):
                if number < self.roulette[j]:
                    self.parent.append(j)
                    break

        # Makes the first value also the last one, used for making cross-over code simpler.
        self.parent.append(self.parent[0])

    def crossover(self, method, num_of_points, offspring_rate, selection, limit):
        points = []
        weight = 0

        # Point method
        if method == "point":
            # Generate points of mutation
            for i in range(num_of_points):
                self.point_gen(points)

            # Generate offspring using 1-point crossover
            for i in range(self.chromosomeSize):
                numb = np.random.rand()
                if numb < offspring_rate:
                    # Generate first parent data
                    self.new_population[i, 0:points[0]] = self.population[self.parent[i], 0:points[0]]
                    # Generate second parent data
                    self.new_population[i, points[0]:-1] = self.population[self.parent[i + 1], points[0]:-1]
                    # If the offspring weight is over 30, regenerate the offspring
                    self.weight_constraint(i, limit, selection)

        # Uniform method - Work in progress
        elif method == "uniform":
            for i in range(self.geneSize):
                pass

        # Improper method selection
        else:
            print("Improper method selected")

    def weight_constraint(self, chromosome_number, limit, selection):
        weight = 0

        for item in range(self.geneSize):
            if self.population[chromosome_number, item] == 1:
                weight = weight + selection[0, item]

        while weight > limit:
            self.reroll_chromosome(chromosome_number)

            for item in range(self.geneSize):
                if self.population[chromosome_number, item] == 1:
                    weight = weight + selection[0, item]

    def point_gen(self, points):
        point = np.random.randint(1, self.geneSize)

        # Loop until unique point is identified
        while point in points:
            point = np.random.randint(1, self.geneSize)

        points.append(point)
        return points

    def mutation(self, rate):
        for i in range(self.chromosomeSize):
            for j in range(self.geneSize):
                # Roll the dice at every bit/gene, if successful, change the gene
                number = np.random.rand()
                if number < rate:
                    if self.population[i, j] == 1:
                        self.population[i, j] = 0
                    else:
                        self.population[i, j] = 1

    def gen_reset(self):
        # New generation is now the current generation, reset new population and add gen number
        self.fitness = []
        self.fitnessPercent = []
        self.roulette = np.zeros(self.chromosomeSize)
        self.parent = []
        self.population = deepcopy(self.new_population)
        self.new_population = np.zeros([self.chromosomeSize, self.geneSize])
        self.generation = self.generation + 1

    """
    Advances Algorithms
    
    This section contains advanced genetic algorithm 
    """


def main():
    # Knapsack problem
    generation = 100
    knapsack = np.array([[5, 18], [15, 15], [2, 8], [10, 7], [19, 12], [13, 10], [4, 12], [5, 5], [9, 12], [3, 2]])
    ga = GeneticAlgorithm(knapsack.shape[0], 15)
    ga.populate(knapsack.T, 30)

    for gen in range(generation):
        ga.fitness_calc(knapsack.T)
        ga.roulette_calc()
        ga.parent_selection()
        ga.crossover('point', 1, 0.7, knapsack.T, 30)
        ga.mutation(0.01)
        print("The average fitness of generation {} is {}, and the best fitness is {}.".format(ga.generation,
                                                                                               np.mean(ga.fitness),
                                                                                               np.max(ga.fitness)))
        ga.gen_reset()

    return ga


if __name__ == "__main__":
    ga = main()
