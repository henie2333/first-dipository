import os
from random import random, randint, seed
import pylab as plt
import numpy as np

"""
遗传算法：
有很多袋鼠，它们降落到喜玛拉雅山脉的任意地方。
这些袋鼠并不知道它们的任务是寻找珠穆朗玛峰。但每过几年，就在一些海拔高度较低的地方射杀一些袋鼠。
于是，不断有袋鼠死于海拔较低的地方，而越是在海拔高的袋鼠越是能活得更久，
也越有机会生儿育女。就这样经过许多年，这些袋鼠们竟然都不自觉地聚拢到了一个个的山峰上，
可是在所有的袋鼠中，只有聚拢到珠穆朗玛峰的袋鼠被带回了美丽的澳洲。
"""

MIN = -1.0
MAX = 2.0
EPS = 0.000001
L = int(np.log2((MAX-MIN)/EPS))+1
P_COUNT = 50
MAX_GEN = 200

def f(x):#目标函数，使用numpy中的函数，以便对数组运算#本例是求目标函数的最大值，在区间[MIN, MAX]上
    return x*np.sin(10*np.pi*x)+2.0


def tobin(x):
    b = bin(int(round((x+1)/(MAX-MIN)*(2**L-1.0))))
    b = b[2:]
    if len(b) < L:
        b = '0'*(L-len(b))+b
    return b


def todec(b):
    x = int(b,2)/(2**L-1.0)*(MAX-MIN)-1
    return x


def individual():
    #生成种群中的一个个体，本例中为一个实数，使用二进制字符串编码
    x = random()*(MAX-MIN)+MIN
    b = tobin(x)
    return b
    

def mutate(individual):
    ind = randint(0, L-1)
    single = (int(individual[ind])+1)%2
    single = str(single)
    return individual[:ind]+single+individual[ind+1:]
    

def crossover(male, female, random_co=0.75):
    if random_co >random():
        pos =randint(1,len(male)-1)
        child1 = male[:pos]+ female[pos:]
        child2 = female[:pos]+ male[pos:]
    else:
        child1 = male
        child2 = female
    return child1, child2
        

def population(count):
    """生成指定数量的一个种群count: 种群中个体的数量"""
    return [individual()for x in range(count)]
    

def fitness(individual):
    #适应度函数，本例中直接使用目标函数#目标函数与应用度函数不是一个概念，本例特殊，两者一致
    x = todec(individual)
    return f(x)
    

def grade(pop):
    #计算种群的得分，一般取种群中各个体的适应度函数值的均值#本例中取最大值。
    #summed = sum([fitness(x) for x in pop])
    # #return summed / (len(pop)*1.0)
    values = [fitness(x)for x in pop]
    maxv = max(values)
    return maxv, values.index(maxv)
    

def evolve(pop, random_crossover=0.75, mutate_rate=0.1):
    #进化过程#轮盘赌方法，或称比例法
    grades = [fitness(x)for x in pop]
    total = sum(grades)
    #适应度总和
    for i in range(1, len(pop)):
        grades[i] += grades[i-1]    #各概率换成累积概率
        grades[i-1] = grades[i-1] / total
    grades[-1] = 1.0
    parents = []
    for k in range(len(pop)):
        r = random()
        for i in range(len(pop)):
            if grades[i] > r:
                parents.append(pop[i])
                break#繁衍
    parents_length = len(parents)
    desired_length = len(pop)
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            if mutate_rate >random():#变异
                male = mutate(male)
                female = mutate(female)
            child1, child2 = crossover(male, female, random_crossover)
            children.append(child1)
            children.append(child2)
    return children
    

def evolve2(pop, group_size=10, random_crossover=0.7, mutate_rate=0.1):
    #进化过程
    #锦标赛方法
    parents =[]
    for k in range(len(pop)):
        group =[pop[randint(0,len(pop)-1)]for j in range(group_size)]
        maxfitness =-999
        for p in group:
            temp = fitness(p)
            if maxfitness < temp:
                maxfitness = temp
                tempp = p
        parents.append(tempp)#繁衍
    parents_length = len(parents)
    desired_length = len(pop)
    children =[]
    while len(children)< desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            if mutate_rate > random():
                male = mutate(male)
                female = mutate(female)
            child1, child2 = crossover(male, female, random_crossover)
            children.append(child1)
            children.append(child2)
    return children
    

def showfig(pop, fname, gen, best_sol, best_fit, temp_sol, temp_fit):
    x = np.arange(MIN, MAX,0.01)
    plt.plot(x, f(x))
    pop = np.array([todec(p)for p in pop])
    plt.plot(pop, f(pop), 'ro')
    plt.text(-0.7, 3.8, u'num of individual：%d'% P_COUNT)
    plt.text(-0.7, 3.6, u'present generation：%d/%d'%(gen, MAX_GEN))
    plt.text(-0.7, 3.4, u'present best solution and suitability：%f, %f'%(temp_sol, temp_fit))
    plt.text(-0.7, 3.2, u'knonwn best solution and suitability：%f, %f'%(best_sol, best_fit))
    plt.savefig(fname)
    plt.clf()
    

if __name__ =='__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    seed(2012)
    P_COUNT = 50
    p = population(P_COUNT)
    fitness_history = [grade(p),]
    all_sols = []
    best_sol = todec(p[fitness_history[-1][1]])
    best_fit = fitness_history[-1][0]
    all_sols.append(best_sol)
    showfig(p,'000.png',0, best_sol, best_fit, best_sol, best_fit)
    j = 0
    for i in range(MAX_GEN):
        j += 1
        p = evolve(p)
        fitness_history.append(grade(p))
        temp_fit = fitness_history[-1][0]
        temp_sol = todec(p[fitness_history[-1][1]])
        all_sols.append(temp_sol)
        if best_fit < temp_fit:
            best_fit = temp_fit
            best_sol = temp_sol
    plt.plot([g[0]for g in fitness_history])
    plt.title('suitability in all generations')
    plt.savefig('allfits.png')
    plt.clf()
    plt.plot(all_sols)
    plt.title('best solution in all generation')
    plt.savefig('allsols.png')
