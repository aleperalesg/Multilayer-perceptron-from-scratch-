import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

class MLP:
	def __init__(self,neuralnet):
		self.neuralnet = neuralnet
		self.loss = []
		self.acc = []

	## forward process to compute gradient descent with backpropagation
	def forward(self):

		a = [self.X] ## output of layers
		z = [None]	 ## weighted sum of layers	

		for i in range(len(self.neuralnet)):

			## calculate weighted sum of layers
			z.append(a[i]@self.neuralnet[i].w + self.neuralnet[i].b)

			## sentence for calculate outputs of layers
			if self.neuralnet[i].func == "sigmoid": 
				a.append(sigmoid[0](z[i+1]))

			elif self.neuralnet[i].func == "relu":
				a.append(relu[0](z[i+1]))


		## get accuracy
		Yp = a[len(a)-1]
		Ypp = np.zeros((self.X.shape[0],1))

		## discriminant function if (Yp > 0.5) -> 1, otherwise -> 0
		for i in range(len(Yp)):
			if Yp[i] < 0.5:
				Ypp[i] = 0
			else:
				Ypp[i] = 1
		
		

		## print and save loss and accuracy return z and a lists 	
		print("Accuracy: ",sum(Ypp == self.Y)/len(Y),"  Loss: ",mse_loss[0](a[len(a)-1],self.Y))
		self.acc.append(sum(Ypp == self.Y)/len(Y))
		self.loss.append(mse_loss[0](a[len(a)-1],self.Y))

		## return z and a lists
		return z,a 

	## backward process to compute gradient descent with backpropagation
	def backward(self,a,z):

		## deltas of layers
		delta = []
		for i in reversed(range(0,len(self.neuralnet))):

			w = self.neuralnet[i].w

			## compute gradient for sigmoid function
			if self.neuralnet[i].func == "sigmoid":
				if i == len(self.neuralnet) - 1:
					## delta of output layer
					delta.insert(0,mse_loss[1](self.Y,a[i+1]) * sigmoid[1](a[i+1]))
				else:
					## delta of hidden layer
					delta.insert(0, delta[0]@self.neuralnet[i+1].w.T * sigmoid[1](a[i+1]))

			## compute gradient for ReLu function
			elif self.neuralnet[i].func == "relu":
				if i == len(self.neuralnet) - 1:
					## delta of output layer
					delta.insert(0,mse_loss[1](self.Y,a[i+1]) * relu[1](a[i+1])) 
				else:
					## delta of hidden layer
					delta.insert(0, delta[0]@self.neuralnet[i+1].w.T * relu[1](a[i+1]))

			## update parameters
			self.neuralnet[i].b = self.neuralnet[i].b - np.mean(delta[0],axis=0,keepdims = True) * self.lr
			self.neuralnet[i].w = self.neuralnet[i].w -  a[i].T@delta[0]* self.lr

	## train neural net 
	def train(self,X,Y,lr,iters = 10000):

		self.X = X 
		self.Y = Y
		self.lr = lr
		for i in range(iters):
			z,a = self.forward()
			self.backward(a,z)

		return self.loss

	
    ## predict set test 
	def predict(self,Xtt,Ytt = False):

		ap = [Xtt] ## outputs of layers
		zp = [None] ## weigthed sum of layers
		## forward process
		for i in range(len(self.neuralnet)):
			zp.append(ap[i]@self.neuralnet[i].w +self.neuralnet[i].b)

			if self.neuralnet[i].func == "sigmoid": 
				ap.append(sigmoid[0](zp[i+1]))

			elif self.neuralnet[i].func == "relu":
				ap.append(relu[0](zp[i+1]))

		
		Yp = ap[len(ap)-1]
		Ypp = np.zeros(Xtt.shape[0])
		## discriminant function if (Yp > 0.5) -> 1, otherwise -> 0
		for i in range(len(Yp)):
			if Yp[i] < 0.5:
				Ypp[i] = 0
			else:
				Ypp[i] = 1
		return Ypp


## class for layers
class nn_layer:
	def __init__(self,n_input,num_neurons,func):
		self.n_input = n_input
		self.num_neurons = num_neurons
		self.func = func

		self.w = np.random.rand(self.n_input,self.num_neurons)
		self.b = np.random.rand(self.num_neurons)


## loss function
mse_loss = ((lambda Y,Ypp: np.mean((Ypp-Y)**2)),
			(lambda Y,Ypp: (Ypp-Y)))

## activation functions
sigmoid  = ((lambda X: 1/(1+np.exp(-X))),
			(lambda X: X*(1-X)))

relu = ((lambda X: np.maximum(0,X)),
		(lambda X: (X > 0).astype(int)))


## learning rate
lr = 0.1 
## make dataset
X,Y = make_circles(random_state=42,n_samples = 500,factor = 0.4,noise = 0.09)
Y = Y[:,np.newaxis]

## size of input features
in_net = X.shape[1] 

## get neural net model
neuralnet = [nn_layer(in_net,6,"relu"),nn_layer(6,1,"sigmoid")] ## list of layers
model = MLP(neuralnet) ## create neural net
model.train(X,Y,lr,5000) ## training neural net

## make a grid for visulization of decision boundary
x1 = np.linspace(-1.5,1.5,200)
x2 = np.linspace(-1.5,1.5,200)
X1,X2 = np.meshgrid(x1,x2)
Xtt = np.column_stack((X1.flatten(),X2.flatten()))
## predict
Ypp = model.predict(Xtt) 

## plot dataset distribution and decision bundary
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Multilayer perceptron')
ax1.scatter(X[Y[:,0] == 0,0], X[Y[:,0] == 0,1],c = "skyblue")
ax1.scatter(X[Y[:,0] == 1,0], X[Y[:,0]  == 1,1],c = "salmon")
ax1.set(xlabel='x1', ylabel='x2')
ax1.set_title("Datset distribution")
ax2.scatter(Xtt[Ypp == 0,0], Xtt[Ypp == 0,1],c = "skyblue")
ax2.scatter(Xtt[Ypp == 1,0],Xtt[Ypp  == 1,1],c = "salmon")
ax2.set_title("Decision boundary of neural net")
ax2.set(xlabel='x1', ylabel='x2')
plt.show()
