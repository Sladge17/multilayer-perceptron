import sys
import os
import time

from mp_predict.class_DSexe import *
from mp_predict.class_MPexe import *
import mp_predict.dump as dump

def dec_predictstat(func):
	def wrap():
		file_name = func()
		print(f"\033[32mCreated prediction file: {file_name}\033[37m")
	return wrap

def main(argv):
	statkey = check_argv(argv)
	DSexe.init_DSexe(argv[0],
					dump.features,
					dump.mean,
					dump.std)
	MPexe.init_MPexe(DSexe.x_exe,
					dump.arch,
					dump.f_act,
					dump.weight)
	MPexe.prediction()
	if statkey:
		error = MPexe.cross_entropy(DSexe.y_exe)
		accuracy = MPexe.get_accuracy(DSexe.y_exe)
		print(f"Prediction error: {round(error, 2)}")
		print(f"Prediction accuracy: {round(accuracy, 2)}%")
	write_prediction()

def check_argv(argv):
	if not len(argv) or len(argv) > 2:
		print("\033[31mNeed only one or two arguments\033[37m")
		exit()
	statkey = 0
	if len(argv) == 1:
		return statkey
	if len(argv) == 2 and argv[1] != "-stat" and argv[1] != "-s":
		print("\033[31mUnknown argument\033[37m")
		exit()
	statkey = 1
	return statkey

@dec_predictstat
def write_prediction():
	file_name = "prediction_{}".\
		format(time.strftime("%d%m%y", time.localtime()))
	with open(file_name, 'w') as file:
		for i in range(MPexe.predict.size):
			file.write(f"{MPexe.predict[i]}\n")
	return file_name

if __name__ == "__main__":
	main(sys.argv[1:])
