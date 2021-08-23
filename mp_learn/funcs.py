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
	derivative = np.ones_like(target)
	derivative[target < 0] = 0
	return derivative

def softmax(target):
	target = np.exp(target)
	softmax = target / np.sum(target)
	return softmax

f = {"sigmoid" : sigmoid,
	"hyptan" : hyptan,
	"relu" : relu}

f_drv = {"sigmoid" : sigmoid_derivative,
		"hyptan" : hyptan_derivative,
		"relu" : relu_derivative}


# x = np.array([0.5, -50, 0, 1], np.float32)
# x = softmax(x)
# print(x)

# y = np.array([1, 0], np.float32)
# error = y @ np.log(softmax(x))
# # error = np.log(softmax(x))
# print(error)