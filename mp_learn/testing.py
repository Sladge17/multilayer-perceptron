import numpy as np
from mp_learn.funcs import *
# from funcs import *

def test_mp(x_test, y_test, arch, f_act, weight):
	predict = predict_test(x_test, weight, arch, f_act)
	accuracy = get_accuracy(predict, y_test)
	return accuracy

def predict_test(x_test, weight, arch, f_act):
	z, x = init_arch(arch, x_test.shape[0])
	forward_propagation(z, x, weight, arch, f_act, x_test)
	return z[-1]

def init_arch(arch, size):
	z = [0] * arch.size
	x = [0] * (arch.size - 2)
	for i in range(arch.size - 1):
		z[i] = np.ones([size, arch[i] + 1], np.float32)
		if i < arch.size - 2:
			x[i] = np.zeros([size, arch[i + 1]], np.float32)
	z[-1] = np.ones([size, arch[-1]], np.float32)
	return z, x

def forward_propagation(z, x, weight, arch, f_act, x_test):
	z[0][:, :-1] = x_test[:, :]
	for j in range(1, arch.size - 1):
		x[j - 1][:, :] = z[j - 1] @ weight[j - 1]
		z[j][:, :-1] = f[f_act](x[j - 1])
	z[-1][:, :] = z[-2] @ weight[-1]
	z[-1][:, :] = softmax(z[-1])

def get_accuracy(predict, target):
	accuracy = 0
	for i in range(predict.shape[0]):
		if np.argmax(predict[i]) == np.argmax(target[i]):
			accuracy += 1
	accuracy = accuracy / predict.shape[0] * 100
	return accuracy