import numpy as np
from mp_learn.funcs import *

class MP:

	@staticmethod
	def init_MP(x_train, y_train,
				arch,
				f_act,
				epochs,
				alpha,
				batch,
				start_velocity):
		# MP.y_train = y_train
		# MP.x_train = x_train
		MP.arch = np.array([x_train.shape[1]] + arch, np.int8)
		MP.f_act = f_act
		MP.epochs = epochs
		MP.alpha = alpha
		MP.batch = batch
		MP.init_datasets(x_train, y_train)
		MP.init_arch(MP.arch, batch)
		MP.border = np.zeros(2, np.int32)
		MP.init_optimizer(start_velocity, MP.weight)
		MP.init_metrics(MP.epochs, MP.y_train)
		MP.xy_train = np.zeros_like(np.concatenate((MP.x_train, MP.y_train),
													axis=1))

	@staticmethod
	def init_datasets(x_train, y_train):
		border = int(x_train.shape[0] * 0.9)
		MP.y_valid = y_train[border:]
		MP.x_valid = x_train[border:]
		MP.y_train = y_train[:border]
		MP.x_train = x_train[:border]

	@staticmethod
	def init_arch(arch, batch):
		MP.z = [0] * arch.size
		MP.x = [0] * (arch.size - 2)
		MP.dx = [0] * (arch.size - 1)
		MP.weight = [0] * (arch.size - 1)
		MP.dw = [0] * (arch.size - 1)
		for i in range(arch.size - 1):
			MP.z[i] = np.ones([batch, arch[i] + 1], np.float32)
			if i < arch.size - 2:
				MP.x[i] = np.zeros([batch, arch[i + 1]], np.float32)
				MP.dx[i] = np.zeros_like(MP.x[i])
			MP.weight[i] = np.zeros([MP.z[i][0].size, arch[i + 1]], np.float64)
			MP.dw[i] = np.zeros_like(MP.weight[i])
		MP.z[-1] = np.ones([batch, arch[-1]], np.float32)
		MP.dx[-1] = np.zeros_like(MP.z[-1])

	@staticmethod
	def init_optimizer(start_velocity, weight):
		MP.velocity = [0] * len(weight)
		MP.accumulator = [0] * len(weight)
		for i in range(len(weight)):
			MP.velocity[i] = np.full(weight[i].shape, start_velocity, np.float32)
			MP.accumulator[i] = np.zeros(weight[i].shape, np.float32)

	@staticmethod
	def init_metrics(epochs, y_train):
		MP.error = np.zeros([2, epochs], np.float32)
		MP.accuracy = np.zeros([2, epochs], np.float32)
		MP.z_epoch = np.zeros_like(y_train, np.float32)

	@staticmethod
	def reinit_weight():
		for i in range(MP.arch.size - 1):
			MP.weight[i][:, :] = (1 / np.sqrt(MP.arch[i]) *\
								np.random.rand(MP.z[i][0].size, MP.arch[i + 1]))

	@staticmethod
	def learning():
		for epoch in range(MP.epochs):
			MP.training()
			MP.prediction(MP.x_train)
			MP.error[0, epoch] = np.sum(MP.cross_entropy(MP.z_epoch, MP.y_train))
			MP.accuracy[0, epoch] = MP.get_accuracy(MP.z_epoch, MP.y_train)
			MP.shuffle_trainds()
			MP.prediction(MP.x_valid)
			MP.error[1, epoch] = np.sum(MP.cross_entropy(MP.z_epoch[MP.border[0] : MP.border[1]], MP.y_valid))
			MP.accuracy[1, epoch] = MP.get_accuracy(MP.z_epoch[MP.border[0] : MP.border[1]], MP.y_valid)

	@staticmethod
	def training():
		for i in range(MP.x_train.shape[0] // MP.batch):
			MP.border[0] = i * MP.batch
			MP.border[1] = (i + 1) * MP.batch
			MP.forward_propagation(MP.x_train)
			MP.back_propagation()
			MP.update_weight_adam()

	@staticmethod
	def forward_propagation(x_dataset):
		MP.z[0][:, :-1] = x_dataset[MP.border[0] : MP.border[1]]
		for j in range(1, MP.arch.size - 1):
			MP.x[j - 1][:, :] = MP.z[j - 1] @ MP.weight[j - 1]
			MP.z[j][:, :-1] = f[MP.f_act](MP.x[j - 1])
		MP.z[-1][:, :] = softmax(MP.z[-2] @ MP.weight[-1])

	@staticmethod
	def back_propagation():
		MP.dx[-1][:, :] = MP.z[-1] - MP.y_train[MP.border[0] : MP.border[1]]
		for j in range(MP.arch.size - 2, 0, -1):
			MP.dw[j][:, :] = MP.z[j].T @ MP.dx[j]
			MP.dx[j - 1][:, :] = (MP.dx[j] @ MP.weight[j].T)[:, :-1] *\
									f_drv[MP.f_act](MP.x[j - 1])
		MP.dw[0][:, :] = MP.z[0].T @ MP.dx[0]

	@staticmethod
	def update_weight_adam():
		for j in range(len(MP.weight)):
			MP.velocity[j][:, :] = 0.9 * MP.velocity[j] +\
									(1 - 0.9) * MP.dw[j]
			MP.accumulator[j][:, :] = 0.9 * MP.accumulator[j] +\
										(1 - 0.9) * (MP.dw[j] ** 2)
			MP.weight[j][:, :] -= MP.velocity[j] *\
									(MP.alpha / np.sqrt(MP.accumulator[j]))

	@staticmethod
	def prediction(x_dataset): ## NEED THINK !!!
		# for i in range(x_dataset.shape[0] // MP.batch):
		# 	MP.border[0] = i * MP.batch
		# 	MP.border[1] = (i + 1) * MP.batch
		# 	MP.forward_propagation(x_dataset)
		# 	MP.z_epoch[MP.border[0] : MP.border[1], :] = MP.z[-1]
		# if x_dataset.shape[0] % MP.batch:
		# 	MP.border[0] = MP.border[1]
		# 	MP.border[1] = x_dataset.shape[0] - MP.border[0]
		# 	MP.forward_propagation_tail(x_dataset)
		# 	MP.z_epoch[MP.border[0] :, :] = MP.z[-1][: MP.border[1]]

		if x_dataset.shape[0] // MP.batch:
			for i in range(x_dataset.shape[0] // MP.batch):
				MP.border[0] = i * MP.batch
				MP.border[1] = (i + 1) * MP.batch
				MP.forward_propagation(x_dataset)
				MP.z_epoch[MP.border[0] : MP.border[1], :] = MP.z[-1]
			if x_dataset.shape[0] % MP.batch:
				MP.border[0] = MP.border[1]
				MP.border[1] = x_dataset.shape[0] - MP.border[0]
				MP.forward_propagation_tail(x_dataset)
				# MP.z_epoch[MP.border[0] :, :] = MP.z[-1][: MP.border[1]]
				MP.z_epoch[MP.border[0] : MP.border[0] + MP.border[1], :] =\
					MP.z[-1][: MP.border[1]]
			return
		MP.border[0] = 0
		MP.border[1] = x_dataset.shape[0]
		MP.forward_propagation_tail(x_dataset)
		MP.z_epoch[: MP.border[1], :] = MP.z[-1][: MP.border[1]]

	@staticmethod
	def forward_propagation_tail(x_dataset):
		MP.z[0][: MP.border[1], :-1] = x_dataset[MP.border[0] :]
		for j in range(1, MP.arch.size - 1):
			MP.x[j - 1][: MP.border[1], :] = MP.z[j - 1][: MP.border[1]] @\
											MP.weight[j - 1]
			MP.z[j][: MP.border[1], :-1] = f[MP.f_act](MP.x[j - 1][: MP.border[1]])
		MP.z[-1][: MP.border[1], :] = softmax(MP.z[-2][: MP.border[1]] @\
											MP.weight[-1])

	@staticmethod
	def cross_entropy(z_epoch, y_dataset):
		cross_entropy = -np.log([z_epoch[i, np.argmax(y_dataset[i])]\
								for i in range(z_epoch.shape[0])])
		return cross_entropy

	@staticmethod
	def get_accuracy(z_epoch, y_dataset):
		accuracy = 0
		for i in range(z_epoch.shape[0]):
			if np.argmax(z_epoch[i]) == np.argmax(y_dataset[i]):
				accuracy += 1
		accuracy = accuracy / z_epoch.shape[0] * 100
		return accuracy

	@staticmethod
	def shuffle_trainds():
		MP.xy_train[:, :] = np.concatenate((MP.x_train, MP.y_train), axis=1)
		np.random.shuffle(MP.xy_train)
		MP.x_train[:, :] = MP.xy_train[:, :-2]
		MP.y_train[:, :] = MP.xy_train[:, -2:]
