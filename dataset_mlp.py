import numpy as np
from sklearn.datasets import make_circles, make_moons
import torch

	## One Hot Encoding
def OneHotEncoding(Y):
	k = np.unique(Y)
	OHE = np.zeros((len(Y),len(k)))
	for i in range(len(k)):
		
		OHE[:,i] = Y[:,0] == i
			

	return OHE



	## make dataset
def make_dataset(samples, t = 'moons'):

	

	if t == 'moon':
		X,Y = make_moons(random_state=42,n_samples = samples,noise = 0.099)

	else:
		X,Y = make_circles(random_state=42,n_samples = samples,factor = 0.4,noise = 0.099)


	Y = Y[:,np.newaxis]
	Y = OneHotEncoding(Y)

	return torch.from_numpy(X).to(torch.float32),torch.from_numpy(Y)
	 
"""

		c_k = 2
		t_jack = torch.zeros(X.shape[0],c_k,c_k)

		for k in range(X.shape[0]):
			jac = torch.zeros(c_k,c_k)
			for i in range(c_k):
				for j in range(c_k):
					if i == j:
						jac[i,j] = X[0,i-1] * (1-X[0,i-1])
					else:
						jac[i,j] = - X[0,i-1] * X[0,j-1]

			t_jack[k,:,:] = jac

		return t_jack
"""