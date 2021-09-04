import sys
import time

import dump
from mp_learn.class_DSexe import *
# from mp_learn.class_MPexe import *

def main(argv):
	dataset = check_argv(argv)
	DSexe.init_DSexe(dataset,
					dump.features,
					dump.mean,
					dump.std)

	print(DSexe.y_exe)

def check_argv(argv):
	if not len(argv) or len(argv) > 1:
		print("\033[31mNeed only one argument (dataset)\033[37m")
		exit()
	return argv[0]




if __name__ == "__main__":
	main(sys.argv[1:])

