import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

class MLP:
	def __init__(self,neuralnet):
		self.neuralnet = neuralnet

	def forward(self):

		a = [self.X]
		z = [None]
		for i in range(len(self.neuralnet)):

			z.append(a[i]@self.neuralnet[i].w + self.neuralnet[i].b)
			if self.neuralnet[i].func == "sigmoid": 
				a.append(sigmoid[0](z[i+1]))

			elif self.neuralnet[i].func == "relu":
				a.append(relu[0](z[i+1]))

		print(mse_loss[0](a[len(a)-1],self.Y))
		return z,a 


	def backward(self,a,z):

		delta = []
		for i in reversed(range(0,len(self.neuralnet))):

			w = self.neuralnet[i].w

			if self.neuralnet[i].func == "sigmoid":
				if i == len(self.neuralnet) - 1:
					## output layer
					delta.insert(0,mse_loss[1](self.Y,a[i+1]) * sigmoid[1](a[i+1]))
				else:

					delta.insert(0, delta[0]@self.neuralnet[i+1].w.T * sigmoid[1](a[i+1]))
			
			elif self.neuralnet[i].func == "relu":
				if i == len(self.neuralnet) - 1:
					## output layer
					delta.insert(0,mse_loss[1](self.Y,a[i+1]) * relu[1](a[i+1]))
				else:
					delta.insert(0, delta[0]@self.neuralnet[i+1].w.T * relu[1](a[i+1]))


			self.neuralnet[i].b = self.neuralnet[i].b - np.mean(delta[0],axis=0,keepdims = True) * self.lr
			self.neuralnet[i].w = self.neuralnet[i].w -  a[i].T@delta[0]* self.lr

	def train(self,X,Y,lr,iters = 10000):

		self.X = X
		self.Y = Y
		self.lr = lr
		for i in range(iters):
			z,a = self.forward()
			self.backward(a,z)

	

	def predict(self,Xtt):

		ap = [Xtt]
		zp = [None]
		for i in range(len(self.neuralnet)):
			zp.append(ap[i]@self.neuralnet[i].w +self.neuralnet[i].b)

			if self.neuralnet[i].func == "sigmoid": 
				ap.append(sigmoid[0](zp[i+1]))

			elif self.neuralnet[i].func == "relu":
				ap.append(relu[0](zp[i+1]))

		
		Yp = ap[len(ap)-1]

	
		Ypp = np.zeros(Xtt.shape[0])

		for i in range(len(Yp)):
			if Yp[i] < 0.5:
				Ypp[i] = 0
			else:
				Ypp[i] = 1
		return Ypp


class nn_layer:
	def __init__(self,n_input,num_neurons,func):
		self.n_input = n_input
		self.num_neurons = num_neurons
		self.func = func

		self.w = np.random.rand(self.n_input,self.num_neurons)
		self.b = np.random.rand(self.num_neurons)

		#print("\n",self.w)
		#print(self.b,"\n")


mse_loss = ((lambda Y,Ypp: -np.sum(Y * np.log(Ypp+np.finfo(float).eps))/len(Ypp)),
		   (lambda Y,Ypp: Ypp-Y))
"""

mse_loss = ((lambda Y,Ypp: np.mean((Ypp-Y)**2)),
			(lambda Y,Ypp: (Ypp-Y)))
"""
######## activation functions
sigmoid  = ((lambda X: 1/(1+np.exp(-X))),
			(lambda X: X*(1-X)))

relu = ((lambda X: np.maximum(0,X)),
		(lambda X: (X > 0).astype(int)))


	


## make dataset
X,Y = make_circles(random_state=42,n_samples = 500,factor = 0.4,noise = 0.09)
Y = Y[:,np.newaxis]

lr = 0.1 # learning rate
in_net = X.shape[1]

x1 = np.linspace(-1.5,1.5,200)
x2 = np.linspace(-1.5,1.5,200)

X1,X2 = np.meshgrid(x1,x2)

Xtt = np.column_stack((X1.flatten(),X2.flatten()))


neuralnet = [nn_layer(in_net,4,"relu"),nn_layer(4,1,"sigmoid")]
lr = 0.01

model = MLP(neuralnet)
model.train(X,Y,lr,100000)
Ypp = model.predict(Xtt)

plt.scatter(Xtt[Ypp == 0,0], Xtt[Ypp == 0,1],c = "skyblue")
plt.scatter(Xtt[Ypp == 1,0],Xtt[Ypp  == 1,1],c = "salmon")
plt.axis("equal")
plt.show()
