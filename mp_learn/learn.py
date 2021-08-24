import numpy as np
from mp_learn.funcs import *

# def init_params(arch, batch):
# 	x = [0] * arch.size
# 	weight = [0] * (arch.size - 1)
# 	dx = [0] * (arch.size)
# 	dw = [0] * (arch.size - 1)
# 	for i in range(arch.size - 1):
# 		x[i] = np.ones([batch, arch[i] + 1], np.float32)

# 		weight[i] = np.random.rand(x[i].size, arch[i + 1])
		
# 		# a = 1 / np.sqrt(arch[i])
# 		# weight[i] = a * np.random.rand(x[i].size, arch[i + 1])

# 		# a = 1 / np.sqrt(arch[i] / 2)
# 		# weight[i] = a * np.random.rand(x[i].size, arch[i + 1])

# 		if i:
# 			dx[i] = np.zeros(x[i].size - 1, np.float32)
# 		dw[i] = np.zeros(weight[i].shape, np.float32)
	
# 	x[-1] = np.ones([batch, arch[-1]], np.float32)
# 	dx[-1] = np.zeros(x[-1].size, np.float32)

# 	return x, weight, dx, dw

# def learn_mp(x_train, y_train, arch, f_act, epochs, alpha, batch):
# 	x, weight, dx, dw = init_params(arch, batch)
# 	error = np.zeros(epochs, np.float32)

# 	for epoch in range(epochs):
			
# 		## forward propagation
# 		x[0][:, :-1] = x_train
# 		x[1][:, :-1] = x[0] @ weight[0]
# 		for j in range(2, arch.size - 1):
# 			x[j][:, :-1] = f[f_act](x[j - 1]) @ weight[j - 1]
# 		x[-1][:, :] = f[f_act](x[-2]) @ weight[-1]

# 		## set error
# 		# error[epoch] -= y_train[i] @ np.log(softmax(x[-1]))
# 		error[epoch] -= y_train @ np.log(softmax_batch(x[-1]))
		
# 		## back propagation
# 		dx[-1][:] = softmax(x[-1]) - y_train[i]
# 		for j in range(arch.size - 2, 0, -1):
# 			dw[j][:] += f[f_act](x[j]).reshape(-1, 1) * dx[j + 1]
# 			dx[j][:] = ((dx[j + 1] @ weight[j].T) * f_drv[f_act](x[j]))[:-1]
# 			# dx[j][:] = ((dx[j + 1] @ weight[j].T) * sigmoid_derivative(x[j]))[:-1]
# 		dw[0][:] += x[0].reshape(-1, 1) * dx[1]

# 		## update weight
# 		for i in range(len(weight)):
# 			# weight[i][:] = weight[i] - dw[i] * alpha
# 			weight[i][:] -= dw[i] * alpha
# 			dw[i][:] = 0

# 	return weight, error


def init_params(arch):
	x = [0] * arch.size
	weight = [0] * (arch.size - 1)
	dx = [0] * (arch.size)
	dw = [0] * (arch.size - 1)
	for i in range(arch.size - 1):
		x[i] = np.ones(arch[i] + 1, np.float32)

		# weight[i] = np.random.rand(x[i].size, arch[i + 1])
		
		a = 1 / np.sqrt(arch[i])
		weight[i] = a * np.random.rand(x[i].size, arch[i + 1])

		# a = 1 / np.sqrt(arch[i] / 2)
		# weight[i] = a * np.random.rand(x[i].size, arch[i + 1])

		if i:
			dx[i] = np.zeros(x[i].size - 1, np.float32)
		dw[i] = np.zeros(weight[i].shape, np.float32)
	x[-1] = np.ones(arch[-1], np.float32)
	dx[-1] = np.zeros(x[-1].size, np.float32)

	return x, weight, dx, dw

def cross_entropy(x, y):
	cross_entropy = -np.log(x[np.argmax(y)])
	return cross_entropy



def learn_mp(x_train, y_train, arch, f_act, epochs, alpha, batch):
	x, weight, dx, dw = init_params(arch)
	error = np.zeros(epochs, np.float32)

	for epoch in range(epochs):
		for i in range(x_train.shape[0]):
			
			## forward propagation
			x[0][:-1] = x_train[i]
			x[1][:-1] = x[0] @ weight[0]
			for j in range(2, arch.size - 1):
				x[j][:-1] = f[f_act](x[j - 1]) @ weight[j - 1]
			x[-1][:] = f[f_act](x[-2]) @ weight[-1]

			## set error
			# error[epoch] -= y_train[i] @ np.log(softmax(x[-1]))
			# error[epoch] -= np.log(softmax(x[-1])[np.argmax(y_train[i])])
			error[epoch] += cross_entropy(softmax(x[-1]), y_train[i])
			
			## back propagation
			dx[-1][:] = softmax(x[-1]) - y_train[i]
			for j in range(arch.size - 2, 0, -1):
				dw[j][:] += f[f_act](x[j]).reshape(-1, 1) * dx[j + 1]
				dx[j][:] = ((dx[j + 1] @ weight[j].T) * f_drv[f_act](x[j]))[:-1]
				# dx[j][:] = ((dx[j + 1] @ weight[j].T) * sigmoid_derivative(x[j]))[:-1]
			dw[0][:] += x[0].reshape(-1, 1) * dx[1]

		## update weight
		for i in range(len(weight)):
			# weight[i][:] = weight[i] - dw[i] * alpha
			weight[i][:] -= dw[i] * alpha
			dw[i][:] = 0

	return weight, error


