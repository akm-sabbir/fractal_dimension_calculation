import math
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
class global_configuration():
	nvar = 8
	beta = 8
	iterations = 5000
	varsize = [0, nvar]
	varmin = -0.5
	varmax = 0.5
	maxit = 30
	npop = 50
	pc = 0.8
	nc = 2*math.ceil(pc*npop/2) # Number of Offsprings (Parnets)
	pm = 0.3                # Mutation Percentage
	nm = math.ceil(pm*npop)      # Number of Mutants
	gamma = 0.05
	mu = 0.02
	lr_model = SGDRegressor(max_iter=30, tol=1e-3)
	lr =make_pipeline(StandardScaler(), lr_model) 
	#SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
	def init(self):
		return