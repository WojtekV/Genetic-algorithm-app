import random
import numpy as np
from fitness_function import fitness_function_1, fitness_function_2
import matplotlib.pyplot as plt
import statistics


class GeneticAlgorithm:
    MIN_SCORE = 0.001

    def __init__(self,
                 population_size=1000,
                 search_min=-100,
                 search_max=100,
                 bin_length=10,
                 crossover_prob=0.6,
                 crossover_type='three_point',
                 problem='min',
                 mutation_type='two_point',
                 mutation_prob=0.01,
                 num_of_epochs=100,
                 selection_type='tournament',
                 percent_of_best=30,
                 number_of_elite=5,
                 tournament_size=5,
                 inversion_prob=0.01,
                 function_nr='1'
                 ):

        self.function_number = function_nr
        self.ff = fitness_function_1
        self.set_function()
        self.population_size = population_size
        self.search_min = search_min
        self.search_max = search_max
        self.chr_length = bin_length
        self.population = []
        self.score = []
        self.problem = problem
        self.crossover_type = crossover_type
        self.next_population = []
        self.number_of_elite = number_of_elite
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_type = selection_type
        self.inversion_prob = inversion_prob
        self.percent_of_best = percent_of_best
        self.mutation_type = mutation_type
        self.num_of_epochs = num_of_epochs
        self.tournament_size = tournament_size
        self.best_score_for_epoch = []
        self.best_sub_for_epoch = []
        self.std_devs = []
        self.means = []
        self.validate_parameters()
        self.init_population()
        self.genetic_algorithm()

    def genetic_algorithm(self):
        f = open('populations.txt', "w")

        for _ in range(self.num_of_epochs):
            self.evaluate_score()
            self.next_population.clear()

            self.best_score_for_epoch.append(self.get_score(self.get_best_sub()))
            self.std_devs.append(statistics.stdev(self.score))
            self.means.append(statistics.mean(self.score))
            self.best_sub_for_epoch.append(self.get_best_sub())
            self.next_population.append(self.get_best_sub())

            self.selection()
            self.population = self.next_population.copy()
            self.mutate_population()
            self.inverse_population()
            f.write(f"{self.get_score(self.get_best_sub())}\n")

        f.close()

    def validate_parameters(self):
        """Validates input parameters to only contain correct values"""
        if self.population_size < 50 or self.population_size > 1000:
            self.population_size = 500
        if self.search_min > self.search_max:
            self.search_min, self.search_max = self.search_max, self.search_min
        if self.chr_length < 5 or self.chr_length > 100:
            self.chr_length = 10
        if self.crossover_type not in ['one_point', 'two_point', 'three_point', 'uniform']:
            self.crossover_type = 'two_point'
        if self.mutation_type not in ['one_point', 'two_point', 'edge']:
            self.mutation_type = 'one_point'
        if self.selection_type not in ['roulette', 'tournament', 'best']:
            self.selection_type = 'roulette'
        if self.num_of_epochs > 10000:
            self.num_of_epochs = 100
        if self.number_of_elite > self.population_size:
            self.number_of_elite = 1

    def set_function(self):
        if self.function_number == '1':
            self.ff = fitness_function_1
        elif self.function_number == '2':
            self.ff = fitness_function_2

    def init_population(self):
        """Creates random chromosomes in population"""
        pop = []

        for _ in range(self.population_size):
            sample = ""

            for _ in range(self.chr_length * 2):
                sample = sample + str(random.randint(0, 1))
            pop.append(sample)
        self.population = pop

    def evaluate_score(self):
        """Calculates score for each subject in population"""
        self.score = []
        if self.problem == 'max':
            for i in range(len(self.population)):
                self.score.append(self.get_score(self.population[i]))
        else:
            for i in range(len(self.population)):
                # if self.get_score(self.population[i]) != 0:
                self.score.append(-self.get_score(self.population[i]))
                # else:
                #     self.score.append(float('inf'))

        if not all([sc > 0 for sc in self.score]):
            self.scale_score()

    def scale_score(self):
        """Scales score to only contain positive values"""
        min_score = min(self.score)
        for i in range(len(self.score)):
            self.score[i] += abs(min_score) + self.MIN_SCORE

    def get_score(self, point):
        """Returns value of fitness function on given point"""
        if self.function_number == '1':
            return fitness_function_1(self.decode_bin(point[:self.chr_length]),
                                      self.decode_bin(point[self.chr_length:]))

        else:
            return fitness_function_2(self.decode_bin(point[:self.chr_length]),
                                      self.decode_bin(point[self.chr_length:]))

    def get_point(self, point):
        """Decodes binary chromosome to list containing x and y coordinate"""
        return [self.decode_bin(point[:self.chr_length]), self.decode_bin(point[self.chr_length:])]

    def get_best_sub(self):
        """Returns best subjects from population and add rest to self.population"""
        if self.number_of_elite == 1:
            return self.population[self.score.index(max(self.score))]
        else:
            pop = self.population.copy()
            score = self.score.copy()
            s, pop = (list(t) for t in zip(*sorted(zip(score, pop))))

            for i in range(len(self.population) - self.number_of_elite - 1, len(self.population) - 1):
                self.next_population.append(pop[i])

            return self.population[self.score.index(max(self.score))]

    def decode_bin(self, number):
        """Returns decoded number (from binary to decimal)"""
        return self.search_min + int(number, 2) * (self.search_max - self.search_min) / (2 ** self.chr_length - 1)

    def selection(self):
        """Creates new population (based on selection type and population size)"""
        if self.selection_type == 'roulette':
            score_sum = sum(self.score)
            percent_prob = []

            for i in range(len(self.population) - 1):
                percent_prob.append(self.score[i] / score_sum)
            percent_prob.append(1 - sum(percent_prob))

            while len(self.next_population) < self.population_size:
                first_chr, second_chr = self.get_two_different_chr_by_prob(percent_prob)
                child_list = self.crossover(first_chr, second_chr)
                self.next_population.append(child_list[0])
                self.next_population.append(child_list[1])

        if self.selection_type == 'tournament':
            while len(self.next_population) < self.population_size:
                parents = random.choices(self.population, k=self.tournament_size)
                parents = sorted(parents, key=lambda sub: self.get_score(sub), reverse=True)
                # self.next_population.append(parents[0])
                # self.next_population.append(parents[1])
                child_list = self.crossover(parents[0], parents[1])
                self.next_population.append(child_list[0])
                self.next_population.append(child_list[1])

        if self.selection_type == 'best':
            pop = self.population.copy()
            score = self.score.copy()
            _, pop = (list(t) for t in zip(*sorted(zip(score, pop), reverse=True)))
            while len(self.next_population) < self.population_size:
                parents = [random.randint(0, math.floor(self.percent_of_best/100 * len(self.population))),
                                     random.randint(0, math.floor(self.percent_of_best/100 * len(self.population)))]
                while parents[0] == parents[1]:
                    parents[1] = random.randint(0, math.floor(self.percent_of_best/100 * len(self.population)))
                child_list = self.crossover(pop[parents[0]], pop[parents[1]])
                self.next_population.append(child_list[0])
                self.next_population.append(child_list[1])

    def get_two_different_chr_by_prob(self, percent_prob_list):
        """Returns two chromosomes from population based on probability"""
        first_chr = np.random.choice(len(self.population), p=percent_prob_list)
        second_chr = first_chr
        while first_chr == second_chr:
            second_chr = np.random.choice(np.arange(len(self.population)), p=percent_prob_list)
        return self.population[first_chr], self.population[second_chr]

    def crossover(self, first, second):
        """Performs crossover on given chromosomes and returns them"""
        if self.crossover_type == 'one_point':
            crossover_point = random.randint(1, self.chr_length - 1)
            return [first[:crossover_point] + second[crossover_point:],
                    second[:crossover_point] + first[crossover_point:]]

        if self.crossover_type == 'two_point':
            crossover_points = [random.randint(1, self.chr_length - 1), random.randint(1, self.chr_length - 1)]
            while crossover_points[0] == crossover_points[1]:
                crossover_points[1] = random.randint(1, self.chr_length - 1)

            crossover_points.sort()
            return [first[:crossover_points[0]] +
                    second[crossover_points[0]:crossover_points[1]] +
                    first[crossover_points[1]:],
                    second[:crossover_points[0]] +
                    first[crossover_points[0]:crossover_points[1]] +
                    second[crossover_points[1]:]]

        if self.crossover_type == 'three_point':
            crossover_points = {random.randint(1, self.chr_length - 1)}
            while len(crossover_points) != 3:
                crossover_points.add(random.randint(1, self.chr_length - 1))

            crossover_points = sorted(list(crossover_points))
            return [first[:crossover_points[0]] +
                    second[crossover_points[0]:crossover_points[1]] +
                    first[crossover_points[1]:crossover_points[2]] +
                    second[crossover_points[2]:],
                    second[:crossover_points[0]] +
                    first[crossover_points[0]:crossover_points[1]] +
                    second[crossover_points[1]:crossover_points[2]] +
                    first[crossover_points[2]:]]

        if self.crossover_type == 'uniform':
            first_child = ''
            second_child = ''

            for i in range(self.chr_length * 2):
                if random.uniform(0, 1) < self.crossover_prob:
                    first_child = first_child + second[i]
                    second_child = second_child + first[i]

                else:
                    first_child = first_child + first[i]
                    second_child = second_child + second[i]

            return [first_child, second_child]

    def mutate_population(self):
        for i in range(1, len(self.population)):
            if random.uniform(0, 1) < self.mutation_prob:
                self.population[i] = self.mutate_sub(self.population[i])

    def mutate_sub(self, sub):
        """Performs mutation on single chromosome"""
        if self.mutation_type == 'one_point':
            mutation_point = random.randint(1, self.chr_length - 1)
            if sub[mutation_point] == '0':
                sub = sub[:mutation_point] + '1' + sub[mutation_point + 1:]
            else:
                sub = sub[:mutation_point] + '0' + sub[mutation_point + 1:]
            return sub

        elif self.mutation_type == 'two_point':
            mutation_point = [random.randint(0, self.chr_length - 1), random.randint(0, self.chr_length - 1)]
            while mutation_point[0] == mutation_point[1]:
                mutation_point[1] = random.randint(0, self.chr_length - 1)

            return self.negate_bit(self.negate_bit(sub, mutation_point[0]), mutation_point[1])

        elif self.mutation_type == 'edge':
            if random.randint(0, 1):
                return self.negate_bit(sub, len(sub) - 1)
            else:
                return self.negate_bit(sub, 0)

    def negate_bit(self, sub, place_to_negate):
        sub = list(sub)
        sub[place_to_negate] = '0' if sub[place_to_negate] == '1' else '1'
        return "".join(sub)

    def inverse_population(self):
        for i in range(1, len(self.population)):
            if random.uniform(0, 1) < self.inversion_prob:
                self.population[i] = self.inverse(self.population[i])

    def inverse(self, sub):
        """Performs inverse on given chromosome"""
        inverse_point = [random.randint(1, self.chr_length - 1), random.randint(1, self.chr_length - 1)]
        while inverse_point[0] == inverse_point[1]:
            inverse_point[1] = random.randint(1, self.chr_length - 1)

        inverse_point.sort()
        return sub[:inverse_point[0]] + sub[inverse_point[1]-1:inverse_point[0]-1:-1] + sub[inverse_point[1]:]


if __name__ == '__main__':
    import math

    ga = GeneticAlgorithm()
    print('Best sub: ', ga.get_best_sub())
    print(ga.get_point(ga.get_best_sub()))
    print(ga.get_score(ga.get_best_sub()))
    plt.plot(np.arange(len(ga.best_score_for_epoch)), ga.best_score_for_epoch)
    plt.title('best_score')
    plt.show()
    plt.plot(np.arange(len(ga.std_devs)), ga.std_devs)
    plt.title('std dev')
    plt.show()
    plt.plot(np.arange(len(ga.means)), ga.means)
    plt.title('mean')
    plt.show()

