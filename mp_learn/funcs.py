import numpy as np

def sigmoid(target):
	sigmoid = 1 / (1 + np.exp(-target))
	return sigmoid

def sigmoid_derivative(target):
	derivative = target * (1 - target)
	return derivative

def hyptan(target):
	hyptan = (np.exp(2 * target) - 1) / (np.exp(2 * target) + 1)
	return hyptan

def hyptan_derivative(target):
	derivative = 1 - np.power(target, 2)
	return derivative

def relu(target):
	relu = np.maximum(target, 0)
	return relu

def relu_derivative(target):
	deriv = (target >= 0).astype(np.float32)
	return deriv

def softmax(target):
	target = np.exp(target)
	softmax = target / np.sum(target, axis=1, keepdims=True)
	return softmax

f = {"sigmoid" : sigmoid,
	"hyptan" : hyptan,
	"relu" : relu}

f_drv = {"sigmoid" : sigmoid_derivative,
		"hyptan" : hyptan_derivative,
		"relu" : relu_derivative}
