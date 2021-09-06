import numpy as np
from mp_learn.funcs import *

class MPexe:

	@staticmethod
	def init_MPexe(x_exe,
				arch,
				f_act,
				weight):
		MPexe.x_exe = x_exe
		MPexe.arch = np.array([MPexe.x_exe.shape[1]] + arch, np.int8)
		MPexe.f_act = f_act
		MPexe.weight = weight
		MPexe.init_arch()
		MPexe.predict = np.zeros(MPexe.x_exe.shape[0], np.unicode)

	@staticmethod
	def init_arch():
		MPexe.z = [0] * MPexe.arch.size
		MPexe.x = [0] * (MPexe.arch.size - 2)
		for i in range(MPexe.arch.size - 1):
			MPexe.z[i] = np.ones([MPexe.x_exe.shape[0], MPexe.arch[i] + 1],
								np.float32)
			if i < MPexe.arch.size - 2:
				MPexe.x[i] = np.zeros([MPexe.x_exe.shape[0], MPexe.arch[i + 1]],
										np.float32)
		MPexe.z[-1] = np.ones([MPexe.x_exe.shape[0], MPexe.arch[-1]],
								np.float32)

	@staticmethod
	def prediction():
		MPexe.forward_propagation()
		MPexe.convert_prediction()

	@staticmethod
	def forward_propagation():
		MPexe.z[0][:, :-1] = MPexe.x_exe
		for j in range(1, MPexe.arch.size - 1):
			MPexe.x[j - 1][:, :] = MPexe.z[j - 1] @ MPexe.weight[j - 1]
			MPexe.z[j][:, :-1] = f[MPexe.f_act](MPexe.x[j - 1])
		MPexe.z[-1][:, :] = softmax(MPexe.z[-2] @ MPexe.weight[-1])

	@staticmethod
	def convert_prediction():
		MPexe.predict[:] = list(map(lambda i: 'M' if np.argmax(i)  == 0\
									else 'B', MPexe.z[-1]))

	@staticmethod
	def cross_entropy(y_exe):
		cross_entropy = -np.log([MPexe.z[-1][i, 0 if y_exe[i] == 'M' else 1]\
								for i in range(y_exe.size)])
		cross_entropy = np.sum(cross_entropy).astype(float)
		return cross_entropy

	@staticmethod
	def get_accuracy(y_exe):
		accuracy = 0
		for i in range(MPexe.predict.size):
			if MPexe.predict[i] == y_exe[i]:
				accuracy += 1
		accuracy = accuracy / MPexe.predict.size * 100
		return accuracy
