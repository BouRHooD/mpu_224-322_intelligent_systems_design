import numpy as np
import tensorflow as tf
import random

class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


class FitnessMax:
    def __init__(self):
        self.values = [0]

"""
Индивид - состоит из 784 хромосом
Хромосома состоит из 2 генов 
"""
class GA:
    def __init__(self,
                 size_population,
                 mutation,
                 max_generation,
                 crossover_per=1):

        self.max_generation = max_generation
        self.mutation = mutation
        self.crossover_per = crossover_per
        self.size_population = size_population
        self.count_chrom = 28*28
        self.len_chrom = 2

        self.maxFitnessValues = []
        self.meanFitnessValues = []
        self.best_ind = []
        self.population = []
        self.generationCounter = 0
        self.list_population = []
        self.statistics_generation = []
        self.loss = []
        self.checkStep = 0
        self.log_step = ''

    def chrom_create(self):
        return np.random.uniform(-1, 1, size=2)

    def individual_create(self):
        """Генерация индивидуума"""
        return Individual([[self.chrom_create() for _ in range(self.count_chrom)]])

    def population_create(self):
        """ Создание популяции"""
        return list([self.individual_create() for _ in range(self.size_population)])

    def fitness(self,input, y_true,  individual: Individual):
        """Функция приспособленности"""
        output = np.dot(input, individual[0])
        bce = tf.keras.losses.BinaryCrossentropy()
        loss = bce(y_true, output).numpy()
        # if loss == 0:
        #     loss = 0.00001
        self.loss.append(loss)
        return 1/(loss+0.00000001)

    @staticmethod
    def __clone(value):
        """Клонирование индивидуума"""
        ind = Individual(value[:])
        ind.fitness.values = value.fitness.values
        return ind

    @staticmethod
    def selTournament(population, p_len):
        offspring = []
        for n in range(p_len):
            i1 = i2 = i3 = 0
            while i1 == i2 or i1 == i3 or i2 == i3:
                i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

            offspring.append(
                max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values))

        return offspring     

    @staticmethod
    def mutator( mutant):
        """Мутация """
        for chromosomes in mutant[0]:
            random_gen = random.randint(0, 1)
            chromosomes[random_gen] = random.triangular(-0.5, 0.5, random.gauss(0, 0.5))
            
    def crossover(self, p1, p2):
        """Скрещивание"""
        children_1 = []
        children_2 = []
        
        for chrom_1, chrom_2 in zip(p1[0], p2[0]):
            
            new_chrom_1 = []
            new_chrom_2 = []
            for gen_1, gen_2 in zip(chrom_1, chrom_2):
                if gen_1<gen_2:
                    new_chrom_1.append(np.array(np.random.uniform(gen_1, gen_2, size=1)[0]))
                    new_chrom_2.append(np.array(np.random.uniform(gen_1, gen_2, size=1)[0]))

                else:
                    new_chrom_1.append(np.array(np.random.uniform(gen_2, gen_1, size=1)[0]))
                    new_chrom_2.append(np.array(np.random.uniform(gen_2, gen_1, size=1)[0]))
                    
            children_1.append(np.array(new_chrom_1))
            children_2.append(np.array(new_chrom_2))
        return Individual([children_1]), Individual([children_2])

    def createGA(self):
        """Создание начальной популяции"""
        self.population = self.population_create()
        self.generationCounter = 0
        # fitnessValues = list(map(lambda x: self.fitness(self.inputs, self.y_true, x), self.population))

    def TrainGA(self, inputs, label):
        """Обучение"""
        self.inputs = inputs
        self.label = label
        generationCounter = self.generationCounter
        population = self.population

        FitnessValues = list(map(lambda x: self.fitness(self.inputs, self.label, x), population))

        for individual, fitnessValue in zip(population, FitnessValues):
            individual.fitness.values = fitnessValue
 
        while generationCounter < self.max_generation:
            # self.list_population.append(population)
            generationCounter += 1

            # Отбор 
            offspring = self.selTournament(population, len(population))
            offspring = list(map(self.__clone, offspring))

            new_offspring = []
            # Селекция 
            for parent1, parent2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_per:
                    child1, child2 = self.crossover(parent1, parent2)
                    new_offspring.append(child1)
                    new_offspring.append(child2)
                else:
                    new_offspring.append(parent1)
                    new_offspring.append(parent2)

            offspring = new_offspring.copy()
            # Мутация
            for mutant in offspring:
                if random.random() < self.mutation:
                    self.mutator(mutant)

            freshFitnessValues = list(map(lambda x: self.fitness(self.inputs, self.label, x), offspring))

            # расчет показателей
            for individual, fitnessValue in zip(offspring, freshFitnessValues):
                individual.fitness.values = fitnessValue
            population[:] = offspring
            fitnessValues = [ind.fitness.values for ind in population]
            maxFitness = max(fitnessValues)
            meanFitness = sum(fitnessValues) / len(population)
            self.maxFitnessValues.append(maxFitness)
            self.meanFitnessValues.append(meanFitness)
            
            best_index = fitnessValues.index(max(fitnessValues))
            self.best_ind = population[best_index]
        self.population = population
        self.generationCounter = 0
        return self.best_ind

