import torch
import seaborn as sns
import matplotlib.pyplot as plt
from dataset_mlp import make_dataset
import torch.nn.functional as F

## Set seaborn interface and a seed
sns.set()
torch.manual_seed(42)

## Class for layers
class flatten_layer:
	def __init__(self,n_input,n_output,func = "sigmoid"):

		self.n_input = n_input
		self.n_output = n_output
		self.func = func

		self.w = torch.linspace(-1, 1, self.n_input*self.n_output).view(self.n_input,self.n_output)
		self.b = torch.rand(self.n_output)

		self.delta_w = torch.zeros_like(self.w)
		self.delta_b = torch.zeros_like(self.b)



## Class for mulltilayer perceptrone
class MLP:
	def __init__(self,neuralnet):
		self.neuralnet = neuralnet

	## forward process
	def forward(self):
		a = [self.X] ## Cutput of layers
		z = [None]	 ## Weighted sum of layers


		for i in range(len(self.neuralnet)):

			## Calculate weighted sum of layers
			z.append(a[i]@self.neuralnet[i].w + self.neuralnet[i].b)

			## Calculate output of oputput layer
			if i == len(self.neuralnet)-1:
				neuralnet[i].a = softmax((z[i+1]),False)
				a.append(neuralnet[i].a)

			## Calculate output of oputput layer
			else:
				if self.neuralnet[i].func == "sigmoid": 
					a.append(sigmoid[0](z[i+1]))

				elif self.neuralnet[i].func == "relu":
					a.append(relu[0](z[i+1]))


		return a,z

	## forward process
	def backward(self,a,z):
		
		## Deltas of layers
		delta = []
	
		## Backward process
		for i in reversed(range(0,len(self.neuralnet))):

			w = self.neuralnet[i].w
			
			## Output layer (softmax function)
			if i == len(neuralnet)-1:
				losss = loss(self.Y,a[i+1],True)
				losss = losss.unsqueeze(1)
				sfd = softmax(a[i+1],True)
				d = torch.matmul(losss,sfd)
				d = d.squeeze(1)
				delta.insert(0,d)
				
			else:

				## Compute gradient for sigmoid function
				if self.neuralnet[i].func == "sigmoid":
					## Delta of hidden layer
					delta.insert(0, delta[0]@self.neuralnet[i+1].w.T * sigmoid[1](a[i+1]))

				## Compute gradient for ReLu function
				elif self.neuralnet[i].func == "relu":
					## Delta of hidden layer
					delta.insert(0, delta[0]@self.neuralnet[i+1].w.T * relu[1](a[i+1]))

			## Update parameters
			self.neuralnet[i].b = self.neuralnet[i].b - torch.mean(delta[0],axis=0,keepdims = True) * self.lr
			self.neuralnet[i].w = self.neuralnet[i].w -  a[i].T@delta[0]* self.lr


	## Train function
	def train(self,X,Y,lr,epochs):
		self.X = X
		self.lr = lr
		self.Y = Y


		for i in range(epochs):
			## Get scores (forward process)
			a,z = self.forward()

			## Get gradients and update parameters (backward process)
			self.backward(a,z)

			## Get accuracy
			aux = a[-1]
			Ypp = outs(aux)
			print(f"Accuracy: {torch.sum(torch.all(Y== Ypp, dim=1).int()) /800} Epoch: {i}") 

	## Predcit function
	def predict(self,X):
		self.X = X
		a, _ = self.forward()
		aux = a[-1]
		return outs(aux)


## Loss function
mse_loss = ((lambda Y,Ypp: torch.mean((Ypp-Y)**2)),
			(lambda Y,Ypp: (Ypp-Y)))

## Activation functions
sigmoid  = ((lambda X: 1/(1+torch.exp(-X))),
			(lambda X: X*(1-X)))

relu = ((lambda X: torch.clamp(X, min=0)),
		(lambda X: (X > 0).to(torch.int)))


## Softmax function
def softmax(X,derivate = False):

	if derivate == False:

		sf = F.softmax(X, dim=1)
		return sf
	else:

		c_k = X.shape[1]
		t_jack = torch.zeros(X.shape[0],c_k,c_k)

		for k in range(X.shape[0]):
			p = X[k,:]
			s = p.unsqueeze(-1)
			jacobian = torch.diagflat(s) - s @ s.transpose(-1, -2)
			t_jack[k,:,:] = jacobian
		return t_jack

## Binary cross-entropy loss function
def loss(Y,Ypp,derivate = False):

	eps=1e-10

	if derivate == True:
		loss = torch.zeros_like(Y)
		
		for i in range(len(Y)):
			d = Ypp[i,:] - Y[i,:]
			loss[i,:] = d 
		
		return loss.to(torch.float32)
	else:

		loss = torch.zeros(Y.shape[0],1)
		for i in range(len(Y)):
			loss[i] = -(Y[i,:]@torch.log(Ypp[i,:]+eps))

		return torch.mean(loss,axis = 0).to(torch.float32)

## Convert Y into a one-hot encoded representation based on the indices of the maximum values in each row.
def outs(Y):

	## Find the indices of maximum values:
	_, max_indices = torch.max(Y, dim=1)

	## Create a zero tensor with the same shape as Y
	result = torch.zeros_like(Y)

	## Set the maximum indices to 1 (one-hot encoding):
	return result.scatter_(1, max_indices.unsqueeze(1), 1)



## Set values 
lr = 0.001
epochs = 1000

## create a dataset
X,Y = make_dataset(800,'moons')


## Set layers 
neuralnet = [flatten_layer(2,6,"relu"),flatten_layer(6,8,"relu"),flatten_layer(8,6,"sigmoid"),flatten_layer(6,4,"relu"),flatten_layer(4,4,"sigmoid"),flatten_layer(4,2,"softmax")]

## get a model 
model = MLP(neuralnet)

## train the model 
model.train(X,Y,lr,epochs)

## create a mesh to see the decision boundary
Xm = torch.linspace(-1.5,2.4,100) 
Ym = torch.linspace(-1,1.5,100) 
x_grid, y_grid = torch.meshgrid(Xm, Xm, indexing="ij")

## predict the mesh 
Xtt = torch.stack((x_grid.flatten(),y_grid.flatten())).T
Ytt = model.predict(Xtt)


## Set plt plot
fig, ax = plt.subplots(1,2)

## Plot data distribution
ax[0].scatter(X[Y[:,0] == 0,0],X[Y[:,0] == 0,1], c = 'salmon')
ax[0].scatter(X[Y[:,0] == 1,0],X[Y[:,0] == 1,1], c = 'skyblue')
ax[0].set_xlabel("X1")
ax[0].set_ylabel("X2")
ax[0].set_title("Data distribution")

## Plot Decision boundary
ax[1].scatter(Xtt[Ytt[:,0] == 0,0],Xtt[Ytt[:,0] == 0,1], c = 'salmon')
ax[1].scatter(Xtt[Ytt[:,0] == 1,0],Xtt[Ytt[:,0] == 1,1], c = 'skyblue')
ax[1].set_xlabel("X1")
ax[1].set_ylabel("X2")
ax[1].set_title("Decision boundary")

plt.show()


