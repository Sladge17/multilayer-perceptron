import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mp_learn.settings as settings
import mp_learn.training as training
import mp_learn.testing as testing

from mp_learn.class_MP import *

def main(argv):
	statkey = check_argv(argv)
	df_train, df_test = get_datasets()
	x_train, y_train = separate_dataset(df_train)
	x_test, y_test = separate_dataset(df_test)
	x_train_mean, x_train_std = get_shiftparams(x_train)
	x_train, y_train = prepare_data(x_train, x_train_mean, x_train_std, y_train)
	x_test, y_test = prepare_data(x_test, x_train_mean, x_train_std, y_test)
	
	MP.init_MP(x_train, y_train,
				settings.arch,
				settings.f_act,
				settings.epochs,
				settings.alpha,
				settings.batch,
				settings.start_velocity)


	MP.reinit_weight()
	MP.learning()

	# print(MP.error[0])
	
	plt.figure(figsize=(18, 10))
	plt.plot(range(settings.epochs), MP.error[0], label='error train')
	plt.plot(range(settings.epochs), MP.accuracy[0], label='accuracy train')
	# plt.plot(range(settings.epochs), MP.error[1], label='error valid', linestyle='--')
	# plt.plot(range(settings.epochs), MP.accuracy[1], label='accuracy train', linestyle='--')
	plt.title('Learning progress')
	plt.xlabel('epochs')
	plt.ylabel('error / accuracy')
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()
	
	exit()
	
	
	
	
	
	
	
	
	arch = np.array([x_train.shape[1]] + settings.arch, np.int8)
	print("Learning multilayer perceptron...")
	weight, error_train, accuracy_train, accuracy_test = learning_mp(x_train, y_train,
																	x_test, y_test,
																	arch)
	print("\033[32mLearning done\033[37m")
	if statkey & 0b1:
		print_stats(error_train, accuracy_train, accuracy_test)
	if statkey & 0b10:
		plot_stats(error_train, accuracy_train)
	write_dumpfile(x_train_mean, x_train_std, weight)
	print("\033[32mCreated dump file: dump.py\033[37m")

def check_argv(argv):
	statkey = 0
	if len(argv) > 2:
		print("\033[31mNeed only one or two arguments\033[37m")
		exit()
	for i in argv:
		if i != "-stat" and i != "-s" and i != "-graph" and i != "-g":
			print("\033[31mUnknown argument\033[37m")
			exit()
		if i == "-stat" or i == "-s":
			statkey |= 0b1
			continue
		if i == "-graph" or i == "-g":
			statkey |= 0b10
			continue
	return statkey

def get_datasets():
	try:
		df = pd.read_csv(settings.dataset, header=None)
	except:
		print("\033[31mDataset not exist\033[37m")
		exit()
	border_test = int(df.shape[0] * (100 - settings.percent_test) * 0.01)
	df_train = df.iloc[:border_test]
	df_test = df.iloc[border_test:]
	return df_train, df_test

def separate_dataset(df_dataset):
	x_dataset = df_dataset[settings.features]
	y_dataset = df_dataset[1]
	return x_dataset, y_dataset

def get_shiftparams(x_train):
	x_train_mean = x_train.describe().T['mean']
	x_train_std = x_train.describe().T['std']
	return x_train_mean, x_train_std

def prepare_data(features, x_train_mean, x_train_std, target):
	features = shift_features(features, x_train_mean, x_train_std).values
	target = get_onehotencoding(target)
	return features, target

def shift_features(features, x_train_mean, x_train_std):
	features = (features - x_train_mean) / x_train_std
	return features

def get_onehotencoding(target):
	target = list(map(lambda x: [1, 0] if x == 'M' else [0, 1], target))
	target = np.array(target, np.int8)
	return target

def learning_mp(x_train, y_train, x_test, y_test, arch):
	accuracy_test = 0
	while accuracy_test < settings.target_accuracy:
		weight, error_train, accuracy_train = training.train_mp(x_train, y_train,
																arch,
																settings.f_act,
																settings.epochs,
																settings.alpha,
																settings.batch,
																settings.start_velocity)
		accuracy_test = testing.test_mp(x_test, y_test, arch, settings.f_act, weight)
	return weight, error_train, accuracy_train, accuracy_test

def plot_stats(error_train, accuracy_train):
	plt.figure(figsize=(18, 10))
	plt.plot(range(settings.epochs), error_train[0], label='error train')
	plt.plot(range(settings.epochs), accuracy_train[0], label='accuracy train')
	plt.plot(range(settings.epochs), error_train[1], label='error valid', linestyle='--')
	plt.plot(range(settings.epochs), accuracy_train[1], label='accuracy train', linestyle='--')
	plt.title('Learning progress')
	plt.xlabel('epochs')
	plt.ylabel('error / accuracy')
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()

def print_stats(error_train, accuracy_train, accuracy_test):
	print("Statistic:")
	print(f"\tNumber of epochs: {settings.epochs}")
	print(f"\tLast train error: {round(float(error_train[0, -1]), 3)}")
	print(f"\tLast valid error: {round(float(error_train[1, -1]), 3)}")
	print(f"\tLast train accuracy: {round(float(accuracy_train[0, -1]), 3)}%")
	print(f"\tLast valid accuracy: {round(float(accuracy_train[1, -1]), 3)}%")
	print(f"\033[33m\tTest accuracy: {round(accuracy_test, 3)}%\033[37m")

def write_dumpfile(x_train_mean, x_train_std, weight):
	with open("dump.py", 'w') as file:
		file.write(f"features = {settings.features}\n")
		file.write(f"arch = {settings.arch}\n")
		file.write(f"f_act = \"{settings.f_act}\"\n")
		file.write(f"x_train_mean = {x_train_mean.tolist()}\n")
		file.write(f"x_train_std = {x_train_std.tolist()}\n")
		file.write(f"weight = [0] * {len(weight)}\n")
		for i in range(len(weight)):
			file.write(f"weight[{i}] = {weight[i].tolist()}\n")

if __name__ == "__main__":
	main(sys.argv[1:])