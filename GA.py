from collections import defaultdict
from collections import deque
import numpy as np
import scipy as sp
import math
from Fitness import Fitness
import random
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
#print(random_generator(-.50,.50))
from complex_data import complex_data
from global_configuration import global_configuration
from functools import partial
import multiprocessing as mp
useRouletteWheelSelection=None
useTournamentSelection=None
useRandomSelection=None 
from plotting_data import experiments
from numba import jit, cuda
from random import randint
def random_generator(start, end):
	range_ = end - start
	randNum = start + random.random()*range_
	return randNum 
	
def Initialization():
	# Mutation Rate
	
	global useRouletteWheelSelection
	global useRandomSelection
	ANSWER = 'Random' #input('Choose selection method: , Genetic Algorithm ,...  Roulette Wheel , Tournament , Random , Roulette Wheel')

	useRouletteWheelSelection = 'Roulette Wheel' if(str.lower(ANSWER) == str.lower('Roulette Wheel')) else None  
	useTournamentSelection = 'Tournament' if(str.lower(ANSWER) == str.lower('Tournament')) else None 
	useRandomSelection = 'Random' if(str.lower(ANSWER) == str.lower('Random')) else None
	if useRouletteWheelSelection is not None:
		beta = 8         # Selection Pressure
	if useTournamentSelection is not None:
		TournamentSize = 3   # Tournamnet Size
	pop = np.tile(complex_data(),[global_configuration.npop])
	costS = list()
	for i in range(0, global_configuration.npop):
		result = []
		for j in range(global_configuration.varsize[0], global_configuration.varsize[1]):
			result.append(random_generator(global_configuration.varmin, global_configuration.varmax))
		pop[i].position = np.array(result)
		pop[i].cost = Fitness(pop[i].position)
		print("Fitness cost:" + str(pop[i].cost))
		costS.append(pop[i].cost)
	costS = np.array(costS)
	pop = sortPopulation(pop)
	WorstCost = getCost(pop, index = len(pop)-1)
	BestSolution = pop[0]
	return (pop, costS, WorstCost, BestSolution)
def sortPopulation(pop):
	pop = sorted(pop, key=lambda x : x.cost, reverse=True )
	return pop

def getSol(pop, index = 0):
	return pop[index]
	
def getCost(pop, index = 0):
	return pop[index].cost

def findBestSolution(currentBest, candidateBest):
	
	best = currentBest if currentBest.cost > candidateBest.cost else candidateBest
	return best 
	
	
#@jit(target="cuda")
def mainFunc(args):
	global useRouletteWheelSelection
	global useRandomSelection
	useRandomSelection = 'Random'
	empty_individual,rest = args 
	pop,  costS, WorstCost, BestSolution = rest
	#nfe = np.zeros((global_configuration.maxit, 1))
	BestCost = [0 for i in range(0,global_configuration.maxit)]
	nfe = [0 for i in range(0,global_configuration.maxit)]
	for it in range(0, global_configuration.maxit):
		P = np.exp(-global_configuration.beta*costS/getCost(pop, index = len(pop) - 1))
		P = P/np.sum(P)
		popc = np.tile(complex_data(), [math.ceil(global_configuration.nc/2), 2])
	
		for k in range(0, math.ceil(global_configuration.nc/2)):
			if useRouletteWheelSelection is not None:	
				i1 = RouletteWheelSelection(P)
				i2 = RouletteWheelSelection(P)
			if useRandomSelection is not None:
				i1 = random.randint(1, global_configuration.npop - 1)
				i2 = random.randint(1, global_configuration.npop - 1)
			p1 = pop[i1]
			p2 = pop[i2]
			popc[k][0].position, popc[k][1].position = Crossover(p1.position,p2.position,global_configuration.gamma,global_configuration.varmin,global_configuration.varmax)
			popc[k][0].cost = Fitness(popc[k][0].position)
			popc[k][1].cost = Fitness(popc[k][1].position)
		popm = np.tile(complex_data(), [global_configuration.nm])
		for k in range(0, global_configuration.nm):
			i = random.randint(0, global_configuration.npop-1)
			p = pop[i]
			popm[k].position = Mutate(p.position, global_configuration.mu, global_configuration.varmin, global_configuration.varmax)
			popm[k].cost = Fitness(popm[k].position)
		pop = np.concatenate((np.concatenate((pop, popc.flatten())), popm))
		sortedPop = sortPopulation(pop)
		print("Best Fitness cost in iteration:" + str(sortedPop[0].cost))
		WorstCost = max(WorstCost, sortedPop[len(sortedPop)-1].cost)
		pop = sortedPop[:global_configuration.npop]
		BestSolution = findBestSolution(BestSolution, pop[0])
		print("Best Point: " + str(BestSolution.position))
		print("Best Cost: " + str(BestSolution.cost))
		BestCost[it] = BestSolution.cost
		#if(BestSolution.cost >=1.75 and BestSolution.cost <=1.79):
		#	break
		nfe[it] = it
	print('Iteration :' + str(it) +': NFE = ' + str(nfe[it]) + ', Best Cost= ' + str(BestCost[it]))
	#boxCountingPlot(BestSolution.position)
	return BestSolution

def pointGenerationFunc(w):
	iterations = global_configuration.iterations
	A1 = np.array([[0, 0], [0, 0.17]])
	A2 = np.array([[0.85, 0.04], [-0.04, 0.85]])
	A3 = np.array([[w[0], w[1]], [w[2], w[3]]])
	A4 = np.array([[w[4], w[5]], [w[6], w[7]]])
	t1 = np.array([[0], [0]])
	t2 = np.array([[0], [1.6]])
	t3 = np.array([[0], [1.6]])
	t4 = np.array([[0],[0.44]])
	p1 = 0.01
	p2 = 0.85
	p3 = 0.07
	p4 = 0.07
	x = [0]
	y = [0]
	y1 = [0]
	v = np.array([[0],[0]])
	for n in range(1, iterations):
		k = random_generator(0,1)
		if (k < p1):
			v= np.dot(A1 , v) + t1
		elif (k < p1+p2):
			v= np.dot(A2 , v) + t2
		elif ( k < p1 + p2 + p3):
			v= np.dot(A3 , v ) + t3
		else:
			v= np.dot(A4 , v) + t4
	# now, go back and define your (x,y) point as elements of the vector v
		x.append(v[0][0])
		y.append(v[1][0])
	return x,y
def pointGenerationFuncVer2(w):
	x = []
	y = []
	x.append(0)
	y.append(0)
	current = 0
	for i in range(1, 50000):

        # generates a random integer between 1 and 100
		z = randint(1, 100)

        # the x and y coordinates of the equations
        # are appended in the lists respectively.

        # for the probability 0.01
		if z == 1:
			x.append(0)
			y.append(0.16 * (y[current]))

            # for the probability 0.85
		if z >= 2 and z <= 86:
			x.append(0.85 * (x[current]) + 0.04 * (y[current]))
			y.append(-0.04 * (x[current]) + 0.85 * (y[current]) + 1.6)

        # for the probability 0.07
		if z >= 87 and z <= 93:
		
			x.append(w[0]* (x[current]) - w[1] * (y[current]))
			y.append(w[2] * (x[current]) + w[3] * (y[current]) + 1.6)

            # for the probability 0.07
		if z >= 94 and z <= 100:
			x.append(w[4] * (x[current]) + w[5] * (y[current]))
			y.append(w[6] * (x[current]) + w[7] * (y[current]) + 0.44)
		current = current + 1
	return (x,y)
	
def boxCountingPlot(w):
	X,Y = pointGenerationFuncVer2(w)
	'''
	startR = 0.5
	resultantPointX = []
	resultantPointY = []
	xMin = np.min(X)
	yMin = np.min(Y)
	X = X - xMin  
	Y = Y - yMin
	scale = np.max([np.max(X), np.max(Y)])*1.1
	
		#I use the factor 1.1 to ensure that, after dividing, all coordinates lie strictly below 1
	X = X*(1./scale)
	Y = Y*(1./scale)
	#predictedY = []
	model = global_configuration.lr.fit(X.reshape(-1,1 ), Y)
	for (x,y) in zip(X,Y):		
		result = global_configuration.lr.predict(np.array([x]).reshape(1,-1))
		#print(result)
		if(np.fabs(result - y) <= startR ):
			resultantPointX.append(x*scale/1.1 + xMin)
			resultantPointY.append(y*scale/1.1 + yMin)
	
	#predictedY.append(result)
	X1 = np.array(resultantPointX)
	Y1 = np.array(resultantPointY)
	#PY = np.array(predictedY)
	#plt.plot(X, Y, color = 'green', marker = '.', markersize = 8)
	#plt.scatter(X, Y,  color='black')
	#plt.plot(X, PY, color='blue', linewidth=3)
	#plt.scatter(X1, Y1, s = 0.2, edgecolor = 'green')
	#plt.savefig('pfern.png')
	#plt.xticks(())
	#plt.yticks(())
	#plt.show()
	'''
	return (X, Y)
	
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]	

def RouletteWheelSelection(P):				
	c=np.cumsum(P)
	r = [random.random() for i in range(0, len(c))]
	i = indices( r<=c, lambda x : x != False)[0]
	return i
#RouletteWheelSelection([4,3,5,7,8,9])

def TournamentSelection(pop,m):					
	nPop=len(pop)
	S=random.randint(0, nPop)
	spop=pop[S]
	scosts= spop.cost
	j=min(scosts)
	i=S[j]
	return i

def vector_random_generator(start, end, size):
	result = np.zeros((size,size))
	for i in range(0,size):
		for j in range(0,size):
			result[i][j] = random_generator(start,end)
	return result
def Crossover(x1, x2, gamma, VarMin, VarMax):
	alpha = vector_random_generator(-gamma, 1 + gamma, len(x1))
	y1=np.dot(alpha, x1) + np.dot((1-alpha),x2)
	#print(y1)
	y2=np.dot(alpha, x2) + np.dot((1-alpha), x1)
	y1=np.clip(y1, VarMin, VarMax)
	y2=np.clip(y2, VarMin, VarMax)
	return (y1, y2)


def Mutate(x, mu, VarMin,VarMax):
	nVar = len(x)			
	nmu = math.ceil(mu*nVar)
	np.random.seed(1)
	j = random.randint(min(nVar,nmu),max(nVar,nmu))
	sigma = 0.1 * (VarMax-VarMin)			
	y=x.copy()
	y = x + sigma*norm.ppf(np.random.rand(len(x)+1,len(x)))[j]
	y = np.clip(y,VarMin, VarMax)
	print(y)
	return y

	
def multiProcessingOps():
	import codecs
	pool = mp.Pool(mp.cpu_count())
	n_cpu = 12
    #partial_function = partial(ops, [start for start in range(420000, 647000, 10000)])
	resultant = []
	for i in range(n_cpu):
		tups = Initialization()
		resultant.append((complex_data(),tups))
	#mainFunc(pop, complex_data(), costS, WorstCost, BestSolution)
	results = pool.map(mainFunc, [datum for datum in resultant])
	pool.close()
	pool.join()
	bestCost = -1
	bestParameters=complex_data()
	bestParameters.cost = -1
	for each in results:
		if bestParameters.cost < each.cost:
			bestParameters = each
	print("Best cost is: " + str(bestParameters.cost))
	X,Y = boxCountingPlot(bestParameters.position)
	experiments(X,Y)
	
	return
if __name__ == '__main__':
	multiProcessingOps()
