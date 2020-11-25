from collections import defaultdict
import math
import numpy as np
import scipy as sp
import random
from matplotlib import pyplot as plt
from global_configuration import global_configuration
from housdroff import cost_function
import pylab as pl
NFE = 0


from sklearn.linear_model import SGDRegressor
from sklearn.linear_model.logistic import _logistic_loss
##################################

def get_rows():
	return 
def iter_minibatches(X, y, y1, chunksize=10000):
	# Provide chunks one by one
	chunkstartmarker = 0
	numtrainingpoints = len(X)
	while chunkstartmarker < numtrainingpoints:
		#chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
		X_chunk, y_chunk, y1_chunk = X[chunkstartmarker:chunkstartmarker+chunksize], y[chunkstartmarker: chunkstartmarker+chunksize], y1[chunkstartmarker: chunkstartmarker+chunksize]
		yield X_chunk, y_chunk, y1_chunk
		chunkstartmarker += chunksize

###########################################################
def random_generator(start, end):
	range_ = end - start
	randNum = start + random.random()*range_
	return randNum
def TBO(w):
	iterations = global_configuration.iterations
	scales = np.logspace(0.01, 1, num=30, endpoint=False, base=2)
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
	#v = np.array([[0],[0]])
	points = []
	current = 0
	for n in range(1, iterations):
		k = random.randint(1,100)
		#k = random_generator(0,1)
		if k == 1:#if (k <= p1):
			#v= np.dot(A1 , v) + t1
			x.append(0)
			y.append(0.16 * (y[current]))
		elif k >= 2 and k <= 86:#(k < p1+p2):
			#v= np.dot(A2 , v) + t2
			x.append(0.85 * (x[current]) + 0.04 * (y[current]))
			y.append(-0.04 * (x[current]) + 0.85 * (y[current]) + 1.6)
		elif k >= 87 and k <= 93:#( k < p1 + p2 + p3):
			#v= np.dot(A3 , v ) + t3
			x.append(w[0]* (x[current]) - w[1] * (y[current]))
			y.append(w[2] * (x[current]) + w[3] * (y[current]) + 1.6)
		elif k >= 94 and k <= 100:
			#v= np.dot(A4 , v) + t4
			x.append(w[4] * (x[current]) + w[5] * (y[current]))
			y.append(w[6] * (x[current]) + w[7] * (y[current]) + 0.44)
		else:
			pass
	# now, go back and define your (x,y) point as elements of the vector v
		#x.append(v[0][0])
		#y.append(v[1][0])
		points.append((x[current],y[current]))#(v[0][0],v[1][0]))
		current += 1
		#y1.append(np.clip(np.random.rand(), -0.05,0.05))
	#plt.plot(x, y, color='green', marker='.', linestyle='dashed',linewidth=2, markersize=12)
	#plt.show()
	#x = np.array(x)
	#y = np.array(y)
	#y1 = np.array(y1)
	#x = x - np.min(x)
	#y = y - np.min(y)
	#y1 = y1 - np.min(y1)
	#I use the factor 1.1 to ensure that, after dividing, all coordinates lie strictly below 1
	#scale = np.max([np.max(x),np.max(y),np.max(y1)])*1.1
	#x = x*(1./scale)
	#y = y*(1./scale)
	#for (X,Y) in zip(x,y):
	#	points.append((X,Y))
	#y1 = y1*(1./scale)
	cost = cost_function(pl.array(points),x,y, scales)
	#A = np.transpose(np.concatenate((x,y)))
	#lr = global_configuration.lr
	#for X, Y, Y1 in iter_minibatches(x, y, y1):
	#	model = lr.fit(X.reshape(-1,1 ), Y)#(np.hstack((X.reshape(-1,1 ), Y.reshape(-1,1 ))), Y1)
	#print("model coefficient" + str(global_configuration.lr_model.coef_))
	#m = _logistic_loss(global_configuration.lr_model.coef_, np.hstack((x.reshape(-1,1), y.reshape(-1,1 ))), y1, 1 / global_configuration.lr_model.C)
	#m = _logistic_loss(global_configuration.lr_model.coef_, x.reshape(-1,1), y1, 1 / global_configuration.lr_model.C)
	return cost
	
def Fitness(x):
	return TBO(x)
	
#Fitness([0.20, -0.26, 0.23, 0.22, -0.15, 0.28, 0.26, 0.24])
