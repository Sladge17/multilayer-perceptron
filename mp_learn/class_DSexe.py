import numpy as np
import pandas as pd

class DSexe:

	@staticmethod
	def init_DSexe(dataset, features, mean, std):
		df = DSexe.get_datasets(dataset)
		mean = np.array(mean, np.float32)
		std = np.array(std, np.float32)
		DSexe.extract_data(df, features, mean, std)

	@staticmethod
	def get_datasets(dataset):
		try:
			df = pd.read_csv(dataset, header=None)
		except:
			print("\033[31mDataset not exist\033[37m")
			exit()
		return df

	@staticmethod
	def extract_data(df, features, mean, std):
		DSexe.x_exe = ((df[features] - mean) / std).values

	@staticmethod
	def get_y_exe(dataset):
		y_exe = DSexe.get_datasets(dataset)[1].values
		return y_exe
