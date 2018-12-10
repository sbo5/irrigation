"""
Genetic Algorithm code.
"""
from __future__ import (absolute_import, print_function, division, unicode_literals)
import numpy as np
import random
import time
from copy import deepcopy
from central_FDM_explicit_para_est_Generic_1D import simulate, objective, ode


class GeneticAlgorithm:

    def __init__(self, gene_size, chromosome_size, orig_parameters, var=0.1):
        self.geneSize = gene_size
        self.chromosomeSize = chromosome_size
        self.OP = orig_parameters
        self.population = np.zeros((self.chromosomeSize, self.geneSize))
        self.new_population = np.zeros([self.chromosomeSize, self.geneSize])
        self.fitness = []
        self.fitnessPercent = []
        self.roulette = np.zeros(chromosome_size)
        self.parent = []
        self.generation = 0
        self.var = var

    def __repr__(self):
        return 'GeneticAlgorithm({}, {})'.format(self.geneSize, self.chromosomeSize)

    def __str__(self):
        return 'GA simulation with {} genes, {} chromosomes in the {}th generation'.format(self.geneSize,
                                                                                           self.chromosomeSize,
                                                                                           self.generation)

    def populate(self):

        for i in range(self.chromosomeSize):
            self.population[i, :] = self.OP

        for i in range(1, self.chromosomeSize):
            self.population[i, 0] = self.OP[0] * self.var * np.random.randn(1) + self.OP[0]
            self.population[i, 1] = self.OP[1] * self.var * np.random.randn(1) + self.OP[1]
            self.population[i, 2] = self.OP[2] * self.var * np.random.randn(1) + self.OP[2]
            self.population[i, 3] = self.OP[3] * self.var * np.random.randn(1) + self.OP[3]
            self.population[i, 4] = self.OP[4] * self.var * np.random.randn(1) + self.OP[4]

        # Constraints
        for i in range(self.chromosomeSize):
            while not 1e-10 < self.population[i, 0] < 1e-2:
                self.population[i, 0] = self.OP[0] * self.var * np.random.randn(1) + self.OP[0]
            while not 0.302 < self.population[i, 1] < 1:
                self.population[i, 1] = self.OP[1] * self.var * np.random.randn(1) + self.OP[1]
            while not 0 < self.population[i, 2] < 0.084:
                self.population[i, 2] = self.OP[2] * self.var * np.random.randn(1) + self.OP[2]

    """
    Evaluation of the fitness of each chromosome.
    
    Selection: Table of all items' weight and its corresponding score in the format: (Item weight, Score)
    Limit: Max weight
    """

    def fitness_calc(self):

        for i in range(self.chromosomeSize):
            # print(i, end=' ', flush=True)
            print(i, end=' ')
            fitness_individual = objective(self.population[i, :])
            self.fitness.append(fitness_individual)

    def reroll_chromosome(self, chromosome_number):
        for i in range(self.geneSize):
            self.population[chromosome_number, i] = np.random.randint(0, 2)

    def roulette_calc(self):
        total_fitness = np.sum(self.fitness)

        for i in range(len(self.fitness)):
            self.fitnessPercent = self.fitness[i] / total_fitness

        try:
            for i in range(len(self.fitness)):
                self.roulette[i] = np.sum(self.fitnessPercent[0:i + 1])

        # The last value
        except IndexError:
            self.roulette[-1] = 1

    def min_roulette_calc(self):
        total_fitness = np.sum(self.fitness)

        self.fitnessPercent = self.fitness / total_fitness
        self.fitnessPercent = 1 / self.fitnessPercent

        total_fitness = np.sum(self.fitnessPercent)

        for i in range(len(self.fitness)):
            self.fitnessPercent[i] = self.fitnessPercent[i] / total_fitness

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

    def crossover(self, method, num_of_points, offspring_rate, elitism=True, random_chrom=0):
        points = []
        num_of_elites = 0

        # Introduce elitism, guarantee new population won't be worse than last
        if elitism is True:
            num_of_elites = int(round(0.1 * self.chromosomeSize))

            # If elitism is turned on, there should be at least one elite
            assert(num_of_elites > 0)

            for i in range(num_of_elites):
                index = self.fitness.index(sorted(self.fitness)[i])
                self.new_population[i] = self.population[index]

        # Introduce random individuals, to add variation in population
        if random_chrom > 0:
            for i in range(num_of_elites, num_of_elites + random_chrom):
                self.new_population[i, 0] = self.OP[0] * self.var * np.random.randn(1) + self.OP[0]
                self.new_population[i, 1] = self.OP[1] * self.var * np.random.randn(1) + self.OP[1]
                self.new_population[i, 2] = self.OP[2] * self.var * np.random.randn(1) + self.OP[2]
                self.new_population[i, 3] = self.OP[3] * self.var * np.random.randn(1) + self.OP[3]
                self.new_population[i, 4] = self.OP[4] * self.var * np.random.randn(1) + self.OP[4]

            self.weight_constraint()

        # Point method
        if method == "point":

            # Generate offspring using 1-point crossover
            for i in range(num_of_elites + random_chrom, self.chromosomeSize):

                # Generate points of mutation
                for j in range(num_of_points):
                    self.point_gen(points)

                numb = np.random.rand()
                if numb < offspring_rate:
                    # Generate first parent data
                    self.new_population[i, 0:points[0]] = self.population[self.parent[i], 0:points[0]]
                    # Generate second parent data
                    self.new_population[i, points[0]:] = self.population[self.parent[i + 1], points[0]:]
                    # If the offspring weight is over 30, regenerate the offspring
                    self.weight_constraint()
                else:
                    self.new_population[i, :] = self.population[i]

        # Uniform method - Work in progress
        elif method == "uniform":
            for i in range(self.geneSize):
                pass

        # Improper method selection
        else:
            print("Improper method selected")

        for i in range(self.chromosomeSize):
            assert(np.sum(self.new_population[i, :]) != 0)

    def weight_constraint(self):

        for i in range(self.chromosomeSize):
            while not 1e-10 < self.new_population[i, 0] < 1e-2:
                self.new_population[i, 0] = self.OP[0] * self.var * np.random.randn(1) + self.OP[0]
                # print(self.new_population[i, 0], "Generating new parameter 1...")
            while not 0.302 < self.new_population[i, 1] < 1:
                self.new_population[i, 1] = self.OP[1] * self.var * np.random.randn(1) + self.OP[1]
                # print(self.new_population[i, 1], "Generating new parameter 2...")
            while not 0 < self.new_population[i, 2] < 0.084:
                self.new_population[i, 2] = self.OP[2] * self.var * np.random.randn(1) + self.OP[2]
                # print(self.new_population[i, 2], "Generating new parameter 3...")
            while not -np.inf < self.new_population[i, 4] < np.inf:
                self.new_population[i, 4] = self.OP[4] * self.var * np.random.randn(1) + self.OP[4]
                # print(self.new_population[i, 2], "Generating new parameter 3...")

    def point_gen(self, points):
        point = np.random.randint(1, self.geneSize)

        # Loop until unique point is identified
        # while point in points:
        #     point = np.random.randint(1, self.geneSize)

        points.append(point)
        return points

    def mutation(self, rate):
        for j in range(self.chromosomeSize):
            for i in range(self.geneSize):
                # Roll the dice at every bit/gene, if successful, change the gene
                number = np.random.rand()
                if number < rate:
                    self.new_population[j, i] = self.new_population[j, i] * np.random.uniform(0.5, 1.5)

        self.weight_constraint()

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
    start_time = time.time()

    generation = 30
    parameters = [2.88889e-6, 0.43, 0.078, 3.6, 1.56]
    # parameters = [3.95214683e-06, 4.21144447e-01, 7.64092952e-02, 2.64694825e+00, 1.56000000e+00]
    ga = GeneticAlgorithm(5, 20, parameters, var=0.1)
    ga.populate()

    for gen in range(generation):
        ga.fitness_calc()
        ga.min_roulette_calc()
        ga.parent_selection()
        ga.crossover('point', 1, 1, elitism=True, random_chrom=2)
        ga.mutation(0.01)
        print("The average fitness of generation {} is {}, and the best fitness is {}.".format(ga.generation,
                                                                                               np.mean(ga.fitness),
                                                                                               np.min(ga.fitness)))
        print("The parameters are: {}".format(ga.population[ga.fitness.index(np.min(ga.fitness))]))

        if gen != (generation - 1):
            ga.gen_reset()
    print('Time elapsed: {:.3f} sec'.format(time.time() - start_time))

    return ga


if __name__ == "__main__":
    Ga = main()
