import numpy as np
from mp_learn.funcs import *
# from funcs import *

def train_mp(x_train, y_train, arch, f_act, epochs, alpha, batch, start_velocity):
	x_train, y_train, x_valid, y_valid = init_datasets(x_train, y_train)
	z, x, weight, dx, dw = init_arch(arch, batch)
	border = np.zeros(2, np.int32)
	velocity, accumulator = init_optimizer(start_velocity, dw)
	error, accuracy, z_epoch = init_metrics(epochs, y_train)
	xy_train = np.zeros_like(np.concatenate((x_train, y_train), axis=1))
	for epoch in range(epochs):
		train_trainds(z, x, weight, dx, dw, arch, f_act, x_train, y_train,\
						batch, border, alpha, velocity, accumulator)
		predict_testds(z, x, weight, arch, f_act, x_train,\
						border, batch, z_epoch)
		error[0, epoch], accuracy[0, epoch] = fill_metrics(z_epoch, y_train)
		shuffle_dataset(x_train, y_train, xy_train)
		predict_validds(z, x, weight, arch, f_act, x_valid,\
						border, batch, z_epoch)
		error[1, epoch], accuracy[1, epoch] =\
			fill_metrics(z_epoch[border[0] : border[1]], y_valid)
	return weight, error, accuracy

def init_datasets(x_train, y_train):
	border = int(x_train.shape[0] * 0.9)
	y_valid = y_train[border:]
	x_valid = x_train[border:]
	y_train = y_train[:border]
	x_train = x_train[:border]
	return x_train, y_train, x_valid, y_valid

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

def init_optimizer(start_velocity, dw):
	velocity = [0] * len(dw)
	accumulator = [0] * len(dw)
	for i in range(len(dw)):
		velocity[i] = np.full(dw[i].shape, start_velocity, np.float32)
		accumulator[i] = np.zeros(dw[i].shape, np.float32)
	return velocity, accumulator

def init_metrics(epochs, y_train):
	error = np.zeros([2, epochs], np.float32)
	accuracy = np.zeros([2, epochs], np.float32)
	z_epoch = np.zeros_like(y_train, np.float32)
	return error, accuracy, z_epoch

def train_trainds(z, x, weight, dx, dw, arch, f_act, x_train, y_train,\
					batch, border, alpha, velocity, accumulator):
	for i in range(x_train.shape[0] // batch):
		border[0] = i * batch
		border[1] = (i + 1) * batch
		forward_propagation(z, x, weight, arch, f_act, x_train, border)
		back_propagation(z, x, weight, dx, dw, arch, f_act, y_train, border)
		update_weight_adam(weight, dw, alpha, velocity, accumulator)

def forward_propagation(z, x, weight, arch, f_act, x_train, border):
	z[0][:, :-1] = x_train[border[0] : border[1]]
	for j in range(1, arch.size - 1):
		x[j - 1][:, :] = z[j - 1] @ weight[j - 1]
		z[j][:, :-1] = f[f_act](x[j - 1])
	z[-1][:, :] = softmax(z[-2] @ weight[-1])

def back_propagation(z, x, weight, dx, dw, arch, f_act, y_train, border):
	dx[-1][:, :] = z[-1] - y_train[border[0] : border[1]]
	for j in range(arch.size - 2, 0, -1):
		dw[j][:, :] = z[j].T @ dx[j]
		dx[j - 1][:, :] = (dx[j] @ weight[j].T)[:, :-1] * f_drv[f_act](x[j - 1])
	dw[0][:, :] = z[0].T @ dx[0]

def update_weight_adam(weight, dw, alpha, velocity, accumulator):
	for j in range(len(weight)):
		velocity[j][:, :] = 0.9 * velocity[j] + (1 - 0.9) * dw[j]
		accumulator[j][:, :] = 0.9 * accumulator[j] + (1 - 0.9) * (dw[j] ** 2)
		weight[j][:, :] -= velocity[j] * (alpha / np.sqrt(accumulator[j]))

def predict_testds(z, x, weight, arch, f_act, x_train, border, batch, z_epoch):
	for i in range(x_train.shape[0] // batch):
		border[0] = i * batch
		border[1] = (i + 1) * batch
		forward_propagation(z, x, weight, arch, f_act, x_train, border)
		z_epoch[border[0] : border[1], :] = z[-1]
	if x_train.shape[0] % batch:
		border[0] = border[1]
		border[1] = x_train.shape[0] - border[0]
		forward_propagation_tail(z, x, weight, arch, f_act, x_train, border)
		z_epoch[border[0] :, :] = z[-1][: border[1]]

def forward_propagation_tail(z, x, weight, arch, f_act, x_train, border):
	z[0][: border[1], :-1] = x_train[border[0] :]
	for j in range(1, arch.size - 1):
		x[j - 1][: border[1], :] = z[j - 1][: border[1]] @ weight[j - 1]
		z[j][: border[1], :-1] = f[f_act](x[j - 1][: border[1]])
	z[-1][: border[1], :] = softmax(z[-2][: border[1]] @ weight[-1])

def fill_metrics(z_epoch, y):
	error = np.sum(cross_entropy(z_epoch, y))
	accuracy = get_accuracy(z_epoch, y)
	return error, accuracy

def cross_entropy(z_epoch, y):
	cross_entropy = -np.log([z_epoch[i, np.argmax(y[i])] for i in range(z_epoch.shape[0])])
	return cross_entropy

def get_accuracy(z_epoch, y):
	accuracy = 0
	for i in range(z_epoch.shape[0]):
		if np.argmax(z_epoch[i]) == np.argmax(y[i]):
			accuracy += 1
	accuracy = accuracy / z_epoch.shape[0] * 100
	return accuracy

def shuffle_dataset(x_train, y_train, xy_train):
	xy_train[:, :] = np.concatenate((x_train, y_train), axis=1)
	np.random.shuffle(xy_train)
	x_train[:, :] = xy_train[:, :-2]
	y_train[:, :] = xy_train[:, -2:]

def predict_validds(z, x, weight, arch, f_act, x_valid, border, batch, z_epoch):
	if x_valid.shape[0] // batch:
		for i in range(x_valid.shape[0] // batch):
			border[0] = i * batch
			border[1] = (i + 1) * batch
			forward_propagation(z, x, weight, arch, f_act, x_valid, border)
			z_epoch[border[0] : border[1], :] = z[-1]
		if x_valid.shape[0] % batch:
			border[0] = border[1]
			border[1] = x_valid.shape[0] - border[0]
			forward_propagation_tail(z, x, weight, arch, f_act, x_valid, border)
			z_epoch[border[0] :, :] = z[-1][: border[1]]
		return
	border[0] = 0
	border[1] = x_valid.shape[0]
	forward_propagation_tail(z, x, weight, arch, f_act, x_valid, border)
	z_epoch[: border[1], :] = z[-1][: border[1]]
