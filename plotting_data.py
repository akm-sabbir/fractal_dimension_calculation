#importing necessary modules
import matplotlib.pyplot as plt
from random import randint
from random import random
# initializing the list

import random


def random_generator(start, end):
    range_ = end - start
    randNum = start + random.random() * range_
    return randNum


def experiments(X,Y):
	#w = [0.42313111, 0.28861532, -0.31544006,  0.06028117, -0.02300912,  0.42088921, 0.14784621,  0.00920648]
	#w = [ 0.09858182, -0.27130943,  0.17492825, -0.26625909 , 0.26204772, -0.16823385,-0.5,-0.29751681]
	#w=[0.27251918, 0.19857277, 0.48711222, 0.5 ,       0.30546002, 0.37812073,0.5,  0.24714126]
# setting first element to 0
	
	plt.scatter(X, Y, s=0.2, edgecolor='green')
	plt.savefig('pfern.png')
	plt.show()

#experiments()