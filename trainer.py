import sys
import time
import matplotlib.pyplot as plt

import mp_learn.settings as settings
from mp_learn.class_Dataset import *
from mp_learn.class_MP import *

def main(argv):
	statkey = check_argv(argv)
	Dataset.init_Dataset(settings.dataset,
						settings.features,
						settings.percent_test)
	MP.init_MP(Dataset.x_train, Dataset.y_train,
				settings.arch,
				settings.f_act,
				settings.epochs,
				settings.alpha,
				settings.batch,
				settings.start_velocity,
				settings.seed)
	print("Learning multilayer perceptron...", end='\r')
	accuracy_test = learning_mp(Dataset.x_test, Dataset.y_test)
	print("\033[32mLearning multilayer perceptron done\033[37m")
	if statkey & 0b1:
		print_stat(accuracy_test)
	write_dumpfile()
	print("\033[32mCreated dump file: dump.py\033[37m")
	if statkey & 0b10:
		write_report()
	if statkey & 0b100:
		draw_graph()

def check_argv(argv):
	statkey = 0
	if len(argv) > 3:
		print("\033[31mNeed only one or two arguments\033[37m")
		exit()
	for i in argv:
		if i != "-stat" and i != "-s" and\
			i != "report" and i != "-r" and\
			i != "-graph" and i != "-g":
			print("\033[31mUnknown argument\033[37m")
			exit()
		if i == "-stat" or i == "-s":
			statkey |= 0b1
			continue
		if i == "-report" or i == "-r":
			statkey |= 0b10
			continue
		if i == "-graph" or i == "-g":
			statkey |= 0b100
			continue
	return statkey

def learning_mp(x_test, y_test):
	time_start = time.time()
	accuracy_test = 0
	while accuracy_test < settings.target_accuracy:
		MP.reinit_weight()
		MP.learning()
		MP.prediction(x_test)
		accuracy_test = MP.get_accuracy(y_test)
		if time.time() - time_start > 10:
			print("\033[31mСalculation error, try again\033[37m")
			exit()
	return accuracy_test

def print_stat(accuracy_test):
	print("Statistic:")
	print(f"\tNumber of epochs: {MP.epochs}")
	print(f"\tLast train error: {round(float(MP.error[0, -1]), 3)}")
	print(f"\tLast valid error: {round(float(MP.error[1, -1]), 3)}")
	print(f"\tLast train accuracy: {round(float(MP.accuracy[0, -1]), 3)}%")
	print(f"\tLast valid accuracy: {round(float(MP.accuracy[1, -1]), 3)}%")
	print(f"\033[33m\tTest accuracy: {round(accuracy_test, 3)}%\033[37m")

def write_report():
	file_name = "report_{}".format(time.strftime("%d%m%y", time.localtime()))
	with open(file_name, 'w') as file:
		file.write("Epoch\tError train\t\tError test\t\tAccur train\t\tAccur test\n")
		file.write(f"{'=' * 66}\n")
		for i in range(MP.epochs):
			file.write(f"{i + 1}\t\t\
{round(MP.error[0, i].astype(float), 3)}\t\t\t\
{round(MP.error[1, i].astype(float), 3)}\t\t\t\
{round(MP.accuracy[0, i].astype(float), 3)}\t\t\t\
{round(MP.accuracy[1, i].astype(float), 3)}\n")

def draw_graph():
	plt.figure(figsize=(18, 10))
	plt.plot(range(settings.epochs), MP.error[0], label='error train')
	plt.plot(range(settings.epochs), MP.accuracy[0], label='accuracy train')
	plt.plot(range(settings.epochs), MP.error[1], label='error valid', linestyle='--')
	plt.plot(range(settings.epochs), MP.accuracy[1], label='accuracy train', linestyle='--')
	plt.title('Learning progress')
	plt.xlabel('epochs')
	plt.ylabel('error / accuracy')
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()

def write_dumpfile():
	with open("dump.py", 'w') as file:
		file.write(f"features = {settings.features}\n")
		file.write(f"arch = {settings.arch}\n")
		file.write(f"f_act = \"{settings.f_act}\"\n")
		file.write(f"x_train_mean = {Dataset.mean.tolist()}\n")
		file.write(f"x_train_std = {Dataset.std.tolist()}\n")
		file.write(f"weight = [0] * {len(MP.weight)}\n")
		for i in range(len(MP.weight)):
			file.write(f"weight[{i}] = {MP.weight[i].tolist()}\n")

if __name__ == "__main__":
	main(sys.argv[1:])
