import sys
import os
import time

from mp_learn.class_DSexe import *
from mp_learn.class_MPexe import *
try:
	import dump
except ModuleNotFoundError:
	try:
		os.system("python3 trainer.py")
		import dump
	except:
		try:
			os.system("python trainer.py")
			import dump
		except:
			print("\033Training file not found\033[37m")
			exit()

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
		accuracy = MPexe.get_accuracy(DSexe.get_y_exe(argv[0]))
		print(f"Prediction accuracy: {accuracy}%")
	file_name = write_prediction()
	print(f"\033[32mCreated prediction file: {file_name}\033[37m")

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

def write_prediction():
	file_name = "prediction_{}".format(time.strftime("%d%m%y", time.localtime()))
	with open(file_name, 'w') as file:
		for i in range(MPexe.predict.size):
			file.write(f"{MPexe.predict[i]}\n")
	return file_name

if __name__ == "__main__":
	main(sys.argv[1:])
