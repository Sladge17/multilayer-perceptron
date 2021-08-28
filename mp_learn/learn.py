import numpy as np
from mp_learn.funcs import *
# from funcs import *

def init_arch(arch, batch):
	z = [0] * arch.size
	x = [0] * (arch.size - 2)
	dx = [0] * (arch.size - 1)
	weight = [0] * (arch.size - 1)
	dw = [0] * (arch.size - 1)
	for i in range(arch.size - 1):
		z[i] = np.ones([batch, arch[i] + 1], np.float32)
		if i < arch.size - 2:
			x[i] = np.zeros([batch, arch[i + 1]], np.float32)
			dx[i] = np.zeros_like(x[i])
		weight[i] = 1 / np.sqrt(arch[i]) * np.random.rand(z[i][0].size, arch[i + 1])
		dw[i] = np.zeros_like(weight[i])
	z[-1] = np.ones([batch, arch[-1]], np.float32)
	dx[-1] = np.zeros_like(z[-1])
	return z, x, weight, dx, dw

def init_metrics(epochs, y_train):
	error = np.zeros(epochs, np.float32)
	accuracy = np.zeros(epochs, np.float32)
	z_epoch = np.zeros_like(y_train, np.float32)
	return error, accuracy, z_epoch

def set_forward_propagation(z, x, weight, arch, f_act, x_train, border):
	z[0][:, :-1] = x_train[border[0] : border[1]]
	for j in range(1, arch.size - 1):
		x[j - 1][:, :] = z[j - 1] @ weight[j - 1]
		z[j][:, :-1] = f[f_act](x[j - 1])
	z[-1][:, :] = softmax_batch(z[-2] @ weight[-1])

def set_back_propagation(z, x, weight, dx, dw, arch, f_act, y_train, border):
	dx[-1][:, :] = z[-1] - y_train[border[0] : border[1]]
	for j in range(arch.size - 2, 0, -1):
		dw[j][:, :] = z[j].T @ dx[j]
		dx[j - 1][:, :] = (dx[j] @ weight[j].T)[:, :-1] * f_drv[f_act](x[j - 1])
	dw[0][:, :] = z[0].T @ dx[0]

def set_update_weight(weight, dw, alpha):
	for j in range(len(weight)):
		weight[j][:, :] -= dw[j] * alpha

def set_forward_propagation_tail(z, x, weight, arch, f_act, x_train, border):
	z[0][: border[1], :-1] = x_train[border[0] :]
	for j in range(1, arch.size - 1):
		x[j - 1][: border[1], :] = z[j - 1][: border[1]] @ weight[j - 1]
		z[j][: border[1], :-1] = f[f_act](x[j - 1][: border[1]])
	z[-1][: border[1], :] = softmax_batch(z[-2][: border[1]] @ weight[-1])

def cross_entropy_batch(z, y):
	cross_entropy = -np.log([z[i, np.argmax(y[i])] for i in range(z.shape[0])])
	return cross_entropy

def get_accuracy(z_epoch, y_train):
	accuracy = 0
	for i in range(z_epoch.shape[0]):
		if np.argmax(z_epoch[i]) == np.argmax(y_train[i]):
			accuracy += 1
	accuracy = accuracy / z_epoch.shape[0] * 100
	return accuracy

def set_shuffle(x_train, y_train, xy_train):
	xy_train[:, :] = np.concatenate((x_train, y_train), axis=1)
	np.random.shuffle(xy_train)
	x_train[:, :] = xy_train[:, :-2]
	y_train[:, :] = xy_train[:, -2:]


def learn_mp(x_train, y_train, arch, f_act, epochs, alpha, batch):

	z, x, weight, dx, dw = init_arch(arch, batch)
	border = np.zeros(2, np.int32)
	
	error, accuracy, z_epoch = init_metrics(epochs, y_train)

	xy_train = np.zeros_like(np.concatenate((x_train, y_train), axis=1))

	for epoch in range(epochs):
		
		## training
		for i in range(x_train.shape[0] // batch):
			border[0] = i * batch
			border[1] = (i + 1) * batch

			set_forward_propagation(z, x, weight, arch, f_act, x_train, border)
			set_back_propagation(z, x, weight, dx, dw, arch, f_act, y_train, border)
			set_update_weight(weight, dw, alpha)
		
		## prediction
		for i in range(x_train.shape[0] // batch):
			border[0] = i * batch
			border[1] = (i + 1) * batch
			set_forward_propagation(z, x, weight, arch, f_act, x_train, border)
			z_epoch[border[0] : border[1], :] = z[-1]
		
		if x_train.shape[0] % batch:
			border[0] = border[1]
			border[1] = x_train.shape[0] - border[0]
			set_forward_propagation_tail(z, x, weight, arch, f_act, x_train, border)
			z_epoch[border[0] :, :] = z[-1][: border[1]]
		
		error[epoch] = np.sum(cross_entropy_batch(z_epoch, y_train))
		accuracy[epoch] = get_accuracy(z_epoch, y_train)

		set_shuffle(x_train, y_train, xy_train)
	
	return weight, error, accuracy