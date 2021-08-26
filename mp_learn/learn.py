import numpy as np
from mp_learn.funcs import *
# from funcs import *

def init_params(arch, batch):
	z = [0] * arch.size

	x = [0] * (arch.size - 2)
	dx = [0] * (arch.size - 1)

	weight = [0] * (arch.size - 1)
	dw = [0] * (arch.size - 1)

	for i in range(arch.size - 1):
		z[i] = np.ones([batch, arch[i] + 1], np.float32)

		# weight[i] = np.random.rand(x[i].size, arch[i + 1])
		
		if i < arch.size - 2:
			x[i] = np.zeros([batch, arch[i + 1]], np.float32)
			# dx[i] = np.zeros(x[i].size, np.float32)
			dx[i] = np.zeros_like(x[i])

		a = 1 / np.sqrt(arch[i])
		weight[i] = a * np.random.rand(z[i][0].size, arch[i + 1])

		# a = 1 / np.sqrt(arch[i] / 2)
		# weight[i] = a * np.random.rand(x[i].size, arch[i + 1])

		# if i:
		# 	dx[i] = np.zeros(z[i].size - 1, np.float32)
	
		# dw[i] = np.zeros(weight[i].shape, np.float32)
		dw[i] = np.zeros_like(weight[i])
	
	z[-1] = np.ones([batch, arch[-1]], np.float32)
	# dx[-1] = np.zeros(z[-1][0].size, np.float32)
	dx[-1] = np.zeros_like(z[-1])

	return z, x, weight, dx, dw

def cross_entropy(z, y):
	cross_entropy = -np.log(z[np.argmax(y)])
	return cross_entropy

def cross_entropy_batch(z, y):
	cross_entropy = -np.log([z[i, np.argmax(y[i])] for i in range(z.shape[0])])
	return cross_entropy



# def learn_mp(x_train, y_train, arch, f_act, epochs, alpha, batch):
	
# 	batch = 1 # <--- tmp string

# 	z, x, weight, dx, dw = init_params(arch, batch)
# 	error = np.zeros(epochs, np.float32)

# 	for epoch in range(epochs):
# 		for i in range(x_train.shape[0]):
			
# 			# ## forward propagation
# 			z[0][:, :-1] = x_train[i]
# 			for j in range(1, arch.size - 1):
# 				x[j - 1][:, :] = z[j - 1] @ weight[j - 1]
# 				z[j][:, :-1] = f[f_act](x[j - 1])
# 			z[-1][:, :] = softmax(z[-2] @ weight[-1])

# 			## set error
# 			error[epoch] += cross_entropy(z[-1][0], y_train[i])


# 			## back propagation
# 			# dx[-1][0][:] = z[-1][0] - y_train[i]
# 			dx[-1][:, :] = z[-1] - y_train[i]
# 			for j in range(arch.size - 2, 0, -1):
# 				dw[j][:, :] += z[j].T @ dx[j]

# 				# dx[j - 1][0][:] = (dx[j][0] @ weight[j].T)[:-1] * f_drv[f_act](x[j - 1][0])
# 				dx[j - 1][:, :] = (dx[j] @ weight[j].T)[:, :-1] * f_drv[f_act](x[j - 1])

# 				# dx[j][:] = ((dx[j + 1] @ weight[j].T) * sigmoid_derivative(x[j]))[:-1]
# 			dw[0][:, :] += z[0].T @ dx[0]

# 		## update weight
# 		for i in range(len(weight)):
# 			# weight[i][:] = weight[i] - dw[i] * alpha
# 			weight[i][:, :] -= dw[i] * alpha
# 			dw[i][:, :] = 0

# 	return weight, error


def learn_mp(x_train, y_train, arch, f_act, epochs, alpha, batch):

	z, x, weight, dx, dw = init_params(arch, batch)
	error = np.zeros(epochs, np.float32)

	z_epoch = np.zeros_like(y_train, np.float32)

	for epoch in range(epochs):

		for i in range(x_train.shape[0] // batch):
			
			# ## forward propagation
			z[0][:, :-1] = x_train[i * batch : (i + 1) * batch]
			for j in range(1, arch.size - 1):
				x[j - 1][:, :] = z[j - 1] @ weight[j - 1]
				z[j][:, :-1] = f[f_act](x[j - 1])
			z[-1][:, :] = softmax_batch(z[-2] @ weight[-1])

			z_epoch[i * batch : (i + 1) * batch, :] = z[-1]


			## back propagation
			# dx[-1][0][:] = z[-1][0] - y_train[i]
			dx[-1][:, :] = z[-1] - y_train[i * batch : (i + 1) * batch]
			for j in range(arch.size - 2, 0, -1):
				dw[j][:, :] = z[j].T @ dx[j]

				# dx[j - 1][0][:] = (dx[j][0] @ weight[j].T)[:-1] * f_drv[f_act](x[j - 1][0])
				dx[j - 1][:, :] = (dx[j] @ weight[j].T)[:, :-1] * f_drv[f_act](x[j - 1])

				# dx[j][:] = ((dx[j + 1] @ weight[j].T) * sigmoid_derivative(x[j]))[:-1]
			dw[0][:, :] = z[0].T @ dx[0]

			## update weight
			for j in range(len(weight)):
				# weight[j][:] = weight[j] - dw[j] * alpha
				weight[j][:, :] -= dw[j] * alpha
				# dw[j][:, :] = 0

		## tail calculation
		if x_train.shape[0] // batch:
			# ## forward propagation
			tail = x_train.shape[0] - (i + 1) * batch
			z[0][: tail, :-1] = x_train[(i + 1) * batch :]
			for j in range(1, arch.size - 1):
				x[j - 1][: tail, :] = z[j - 1][: tail] @ weight[j - 1]
				z[j][: tail, :-1] = f[f_act](x[j - 1][: tail])
			z[-1][: tail, :] = softmax_batch(z[-2][: tail] @ weight[-1])

			z_epoch[(i + 1) * batch :, :] = z[-1][: tail]


		## set error
		# error[epoch] += cross_entropy(z[-1][0], y_train[i])
		# error[epoch] += np.sum(cross_entropy_batch(z[-1], y_train))
		error[epoch] += np.sum(cross_entropy_batch(z_epoch, y_train))

	return weight, error


# arr = np.array([10, 4, 3, 2])
# z, x, weight, dx, dw = init_params(np.array([10, 4, 3, 2]))
# print(len(z[1]))
