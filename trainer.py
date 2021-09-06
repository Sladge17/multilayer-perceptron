import sys
import time
import matplotlib.pyplot as plt

import mp_learn.settings as settings
from mp_learn.class_DS import *
from mp_learn.class_MP import *

def dec_learnstat(func):
	def wrap():
		print("Learning multilayer perceptron...", end='\r')
		func()
		print("\033[32mLearning multilayer perceptron done\033[37m")
		time.sleep(1)
	return wrap

def dec_dumpstat(func):
	def wrap():
		func()
		print("\033[32mCreated dump file: dump.py\033[37m")
	return wrap

def dec_reportstat(func):
	def wrap():
		file_name = func()
		print(f"\033[32mCreated report file: {file_name}\033[37m")
	return wrap

def main(argv):
	statkey = check_argv(argv)
	DS.init_DS(settings.dataset,
				settings.features,
				settings.percent_test)
	MP.init_MP(DS.x_train, DS.y_train,
				DS.x_test, DS.y_test,
				settings.arch,
				settings.f_act,
				settings.epochs,
				settings.alpha,
				settings.batch,
				settings.start_velocity,
				settings.seed)
	learning_mp()
	if statkey & 0b1:
		print_stat()
	if statkey & 0b10:
		write_report()
	write_dumpfile()
	if statkey & 0b100:
		draw_graph()

def check_argv(argv):
	statkey = 0
	if len(argv) > 3:
		print("\033[31mNeed arguments between zero and three\033[37m")
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

@dec_learnstat
def learning_mp():
	time_start = time.time()
	while MP.accuracy[1, -1] < settings.target_accuracy:
		MP.reinit_weight()
		MP.learning()
		if time.time() - time_start > 10:
			print("\033[31mСalculation error, try again\033[37m")
			exit()

def print_stat():
	print("Statistics:")
	print(f"\tNumber of epochs: {MP.epochs}")
	print(f"\tLast train error: {round(float(MP.error[0, -1]), 2)}")
	print(f"\tLast test error: {round(float(MP.error[1, -1]), 2)}")
	print(f"\tLast train accuracy: {round(float(MP.accuracy[0, -1]), 2)}%")
	print(f"\033[33m\tLast test accuracy: {round(float(MP.accuracy[1, -1]), 2)}%\033[37m")

@dec_reportstat
def write_report():
	file_name = "report_{}".format(time.strftime("%d%m%y", time.localtime()))
	with open(file_name, 'w') as file:
		file.write("Epoch\tError train\t\tError test\t\tAccur train\t\tAccur test\n")
		file.write(f"{'=' * 66}\n")
		for i in range(MP.epochs):
			file.write(f"{i + 1:5}\t\
{round(MP.error[0, i].astype(float), 2):11}\t\t\
{round(MP.error[1, i].astype(float), 2):10}\t\t\
{round(MP.accuracy[0, i].astype(float), 2):10}%\t\t\
{round(MP.accuracy[1, i].astype(float), 2):9}%\n")
	return file_name

def draw_graph():
	plt.figure(figsize=(18, 10))
	plt.plot(range(settings.epochs), MP.error[0], label='error train')
	plt.plot(range(settings.epochs), MP.accuracy[0], label='accuracy train')
	plt.plot(range(settings.epochs), MP.error[1], label='error test', linestyle='--')
	plt.plot(range(settings.epochs), MP.accuracy[1], label='accuracy test', linestyle='--')
	plt.title('Learning progress')
	plt.xlabel('epochs')
	plt.ylabel('error / accuracy')
	plt.legend(loc='upper right')
	plt.grid()
	plt.show()

@dec_dumpstat
def write_dumpfile():
	with open("dump.py", 'w') as file:
		file.write(f"features = {settings.features}\n")
		file.write(f"mean = {DS.mean.tolist()}\n")
		file.write(f"std = {DS.std.tolist()}\n")
		file.write(f"arch = {settings.arch}\n")
		file.write(f"f_act = \"{settings.f_act}\"\n")
		file.write(f"weight = [0] * {len(MP.weight)}\n")
		for i in range(len(MP.weight)):
			file.write(f"weight[{i}] = {MP.weight[i].tolist()}\n")

if __name__ == "__main__":
	main(sys.argv[1:])
