import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

def model_prices():
	data = pd.read_csv("data.csv")
	labels = data['price']
	train1 = data.drop(['price'],axis=1)
	x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.10,       random_state=2)
	clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
		  learning_rate=0.1, loss='ls')
	clf.fit(x_train, y_train)
	download_dir = os.environ.get('DOWNLOAD_DIR', '.')
	file_pkl = os.path.join(download_dir, 'model.pkl')
	pickle.dump(clf, open(file_pkl,'wb'))
	
if __name__ == '__main__':
   model_prices()
