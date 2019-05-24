from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def cost(params, Y, R, num_features, learning_rate):
	Y = np.matrix(Y)  # (1682, 943)
	R = np.matrix(R)  # (1682, 943)
	num_movies = Y.shape[0]
	num_users = Y.shape[1]
	
	# reshape the parameter array into parameter matrices
	X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
	Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)
	
	# initializations
	J = 0
	X_grad = np.zeros(X.shape)  # (1682, 10)
	Theta_grad = np.zeros(Theta.shape)  # (943, 10)
	
	# compute the cost
	error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
	squared_error = np.power(error, 2)  # (1682, 943)
	J = (1. / 2) * np.sum(squared_error)
	
	# add the cost regularization
	J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
	J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))
	
	# calculate the gradients with regularization
	X_grad = (error * Theta) + (learning_rate * X)
	Theta_grad = (error.T * X) + (learning_rate * Theta)
	
	# unravel the gradient matrices into a single array
	grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
	
	return J, grad

data = loadmat('ex8_movies.mat')
data
Y = data['Y']
R = data['R']
Y.shape, R.shape
	
movie_idx = {}
f = open('movie_ids.txt')
for line in f:
	tokens = line.split(' ')
	tokens[-1] = tokens[-1][:-1]
	movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

params_data = loadmat('ex8_movieParams.mat')
X = params_data['X']
Theta = params_data['Theta']
X.shape, Theta.shape
ratings = np.zeros((1682, 1))

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

Y = np.append(Y, ratings, axis=1)
R = np.append(R, ratings != 0, axis=1)
R = data['R']
Y = data['Y']
movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
Y.shape, R.shape, ratings.shape
features = 10
learning_rate = 10.
X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((X.flatten(), Theta.flatten()), axis = 0)
X.shape, Theta.shape, params.shape

Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))
# J, grad = cost(params, Y_sub, R_sub, features, 1.5)
# J, grad

for i in range(movies):
	idx = np.where(R[i,:] == 1)[0]
	Ymean[i] = Y[i,idx].mean()
	Ynorm[i,idx] = Y[i,idx] - Ymean[i]

Ynorm.mean()

fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate), 
				method='CG', jac=True, options={'maxiter': 100})
fmin

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

X.shape, Theta.shape
predictions = X * Theta.T 
my_preds = predictions[:, -1] + Ymean
my_preds.shape
sorted_preds = np.sort(my_preds, axis=0)[::-1]
sorted_preds[:10]

idx = np.argsort(my_preds, axis=0)[::-1]
idx

print("Top 10 movie predictions:")
for i in range(10):
	j = int(idx[i])
	print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))