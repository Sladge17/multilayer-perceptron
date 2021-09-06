import numpy as np
import pandas as pd

class DS:

	@staticmethod
	def init_DS(dataset, features, percent_test):
		df_train, df_test = DS.get_datasets(dataset, percent_test)
		x_train, y_train = DS.split_dataset(df_train, features)
		x_test, y_test = DS.split_dataset(df_test, features)
		DS.mean = x_train.describe().T['mean']
		DS.std = x_train.describe().T['std']
		DS.x_train, DS.y_train = DS.extract_data(x_train, y_train)
		DS.x_test, DS.y_test = DS.extract_data(x_test, y_test)

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
	def split_dataset(df_dataset, features):
		x_dataset = df_dataset[features]
		y_dataset = df_dataset[1]
		return x_dataset, y_dataset

	@staticmethod
	def extract_data(x, y):
		x = ((x - DS.mean) / DS.std).values
		y = np.array(list(map(lambda i: [1, 0] if i == 'M' else [0, 1], y)),
						np.int8)
		return x, y
