import numpy as np
import pandas as pd

class Dataset:

	@staticmethod
	def init_Dataset(dataset, features, percent_test, *scale_params):
		df_train, df_test = Dataset.get_datasets(dataset, percent_test)
		x_train, y_train = Dataset.separate_dataset(df_train, features)
		x_test, y_test = Dataset.separate_dataset(df_test, features)
		if not len(scale_params):
			Dataset.mean = x_train.describe().T['mean']
			Dataset.std = x_train.describe().T['std']
		else:
			Dataset.mean = scale_params[0]
			Dataset.std = scale_params[1]
		Dataset.x_train, Dataset.y_train =\
			Dataset.prepare_data(x_train, y_train)
		Dataset.x_test, Dataset.y_test =\
			Dataset.prepare_data(x_test, y_test)

	@staticmethod
	def get_datasets(dataset, percent_test):
		try:
			df = pd.read_csv(dataset, header=None)
		except:
			print("\033[31mDataset not exist\033[37m")
			exit()
		border_test = int(df.shape[0] * (100 - percent_test) * 0.01)
		df_train = df.iloc[:border_test]
		df_test = df.iloc[border_test:]
		return df_train, df_test

	@staticmethod
	def separate_dataset(df_dataset, features):
		x_dataset = df_dataset[features]
		y_dataset = df_dataset[1]
		return x_dataset, y_dataset

	@staticmethod
	def prepare_data(x, y):
		x = ((x - Dataset.mean) / Dataset.std).values
		y = np.array(list(map(lambda i: [1, 0] if i == 'M' else [0, 1], y)),
						np.int8)
		return x, y
