import pandas as pd
import numpy as np
import pprint, os

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import grid_search
from sklearn import datasets




def writeResult(model):
	print ("Writng summary... ")
	os.path.join('data/table_sante/', 'svm_summary.txt')
	file = open ('data/table_sante/svm_summary.txt', 'a')
	file.write('Summary: \n')
	file.write('\n')

	file.write('All the combinations of parameters: \n')
	file.write(str(model.get_params))
	file.write('\n')
	file.write('\n')

	file.write('All the scores in differnt combinations of parameters: \n')
	for item in model.grid_scores_:
		file.write('%s\n' %str(item))
	file.write('\n')

	file.write('Best estimator: \n')
	file.write(str(model.best_estimator_))
	file.write('\n')
	file.write('\n')

	file.write('Best score: \n')
	file.write(str(model.best_score_))
	file.write('\n')
	file.write('\n')

	file.write('Best parameters: \n')
	file.write(str(model.best_params_))
	file.write('\n')
	file.write('\n')

	file.write('Prediction of Class: \n')
	pred = model.predict(iris.data)
	os.path.join('data/table_sante/', 'svm_best_pred.txt')
	np.savetxt('data/table_sante/svm_best_pred.txt', pred)
	file.write(str(pred))
	file.write('\n')
	file.write('\n')

	file.write('Probability of prediction: \n')
	proba = model.predict_proba(iris.data)
	os.path.join('data/table_sante/', 'svm_best_proba.txt')
	np.savetxt('data/table_sante/svm_best_proba.txt', proba)
	file.write(str(proba))
	file.write('\n')
	file.write('\n')



if  __name__ == '__main__':
	print ("Getting data... ")
	iris = datasets.load_iris()

	print ("GrideSearchCV defining... ")
	params = [
	{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
	{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
	]
	svc = SVC(probability=True)
	model = grid_search.GridSearchCV(svc, params, verbose=1, n_jobs=-1, cv=50)

	print ("Model testing... ")
	model.fit(iris.data, iris.target)
	writeResult(model)
	print ("Writing done")