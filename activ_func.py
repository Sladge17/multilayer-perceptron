import numpy as np

# def sigmoid(target):
# 	sigmoid = 1 / (1 + np.exp(-target))
# 	return sigmoid

def sigmoid_derivative(target):
	derivative = target * (1 - target)
	return derivative

def softmax(target):
	softmax = target / np.sum(target)
	return softmax
