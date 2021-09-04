import numpy as np
import pandas as pd

class DSexe:

	@staticmethod
	def init_DSexe(dataset, features, mean, std):
		df = DSexe.get_datasets(dataset)
		x_exe, y_exe = DSexe.split_dataset(df, features)
		mean = np.array(mean, np.float32)
		std = np.array(std, np.float32)
		DSexe.prepare_data(x_exe, y_exe, mean, std)

	@staticmethod
	def get_datasets(dataset):
		try:
			df = pd.read_csv(dataset, header=None)
		except:
			print("\033[31mDataset not exist\033[37m")
			exit()
		return df

	@staticmethod
	def split_dataset(df_dataset, features):
		x_dataset = df_dataset[features]
		y_dataset = df_dataset[1]
		return x_dataset, y_dataset

	@staticmethod
	def prepare_data(x_exe, y_exe, mean, std):
		DSexe.x_exe = ((x_exe - mean) / std).values
		DSexe.y_exe = np.array(list(map(lambda i: [1, 0] if i == 'M'
										else [0, 1], y_exe)),
								np.int8)
