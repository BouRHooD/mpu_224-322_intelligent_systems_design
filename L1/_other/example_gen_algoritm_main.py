# Генетические алгоритмы
# Урок 1: https://www.youtube.com/watch?v=52g2Qi_4X3E
# Урок 2: https://www.youtube.com/watch?v=IxFyw4qFytA&t
# Урок 3: https://youtu.be/ond8h5NqtGQ?t=760 
'''
Задача коммивояжёра, !упорядочное скрещивание для маршрутов в графе!, !мутация обменов для упорядоченных списков!
Элитизм - отбираем небольшое число наиболее приспособленных особей
'''
# Урок 4: https://www.youtube.com/watch?v=z9dCWHWcF8c

########################################################################################################################
'''
Одна хромосома описывает все возможные маршруты, список в хромосоме определяет кратчайший маршрут из вершины
'''
########################################################################################################################

# Загружаем библиотеки
import random
import matplotlib.pyplot as plt

# Константы задачи
ONE_MAX_LENGTH = 100     # Длина полежащей оптимизации битовой строки

# Константы генетического алгоритма
POPULATION_SIZE = 200    # Кол-во индивидуумов в популяции
P_CROSSOVER = 0.9        # Вероятность скрещивания
P_MUTATION = 0.1         # Вероятность мутации
MAX_GENERATIONS = 50     # Максимальное кол-во поколений

# Настройки рандома
#RANDOM_SEED = 55         # Зерно для того, чтобы рандом всегда был одним и тем же
#random.seed(RANDOM_SEED) # Присваиваем зерно для рандома

class FitnessMax():
    '''
    Значения приспособленности особи\n
    '''
    def __init__(self):
        self.values = 0

class Individual(list):
    '''
    Особь\n
    Каждое решение – элемент популяции – называется особью
    '''
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()

def oneMaxFitness(individual):
    '''
    Функция приспособленности (англ. fitness function)\n
    Механизм оценки приспособленности каждой особи к текущим условиями окружающей среды\n
    Находим сумму особи - вычисляем приспособленность\n
    Формула fitness=Σ0;N-1(individ[i])
    '''
    return sum(individual)

def individualCreator():
    '''Создаем особь'''
    random_values_list = [random.randint(0,1) for i in range(ONE_MAX_LENGTH)]
    return Individual(random_values_list)

def populationCreator(n=0):
    '''Создаем популяцию из n особей'''
    return list([individualCreator() for i in range(n)])

def clone(value):
    # Создаем хромосому на основе списка (value[:] - делает копию списка)
    ind = Individual(value[:])
    ind.fitness.values = value.fitness.values
    return ind

def selTournament(population, p_len):
    '''Турнирный отбор'''
    offspring = []
    for n in range(p_len):
        # Получаем 3 рандомных НЕ одинаковых индекса
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)
        p1, p2, p3 = population[i1], population[i2], population[i3]
        max_offspring = max([p1, p2, p3], key=lambda ind: ind.fitness.values)
        offspring.append(max_offspring)
    return offspring

def cxOnePoint(child1, child2):
    '''Одноточечный кроссинговер (single-point crossover)'''
    s = random.randint(2, len(child1) - 3)
    cut_child_1 = child1[s:]
    cut_child_2 = child2[s:]
    child1[s:], child2[s:] = cut_child_2, cut_child_1

def mutationFlipBit(mutant, indpd=0.01):
    '''Мутация отдельного гена (инверсия бита)'''
    for index in range(len(mutant)):
        if random.random() < indpd:
            mutant[index] = 0 if mutant[index] == 1 else 1

# Создаем начальную популяцию
population = populationCreator(n=POPULATION_SIZE)
generationCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []
meanFitnessValues = []
vals = []

fitnessValues = [ind.fitness.values for ind in population]
# До тех пор пока не найдем лучшее решенее или не пройдем все поколения
while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    # * Отбираем лучшие особи путем турнира
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))

    # * Выполняем скрещивание неповторяющихся пар родителей 
    offspring_even, offspring_odd = offspring[::2], offspring[1::2]
    for child1, child2 in zip(offspring_even, offspring_odd):
        # Если меньше вероятности скрещивания, то скрещиваем
        # Если срабатывает условие, то родители становятся потомками, иначе они остаются родителями в популяции
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    # * Выполняем мутацию
    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutationFlipBit(mutant, 1.0/ONE_MAX_LENGTH)

    # * Обновляем значение приспособленности новой популяции
    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    # * Обновляем список популяции
    population[:] = offspring

    # * Обновляем список значений приспособленности каждой особи в популяции
    fitnessValues = [ind.fitness.values for ind in population]

    # * Формируем статистику
    vals.append(fitnessValues)

    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len (fitnessValues)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс. приспособ. = {maxFitness}, Средняя присособ. = {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")

'''
# * Выводим собранную статистику в виде графиков
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
'''

# * Интерактивный вывод статистики приспособленности
import time
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(vals[0], ' o', markersize=1)
ax.set_ylim(40, 110)
for v in vals:
    line.set_ydata(v)
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.4)
plt.ioff()
plt.show()