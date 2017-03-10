import pandas as pd
import numpy as np
import pprint, os

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import grid_search

"""
print ("1. Reading data... \n")
df = pd.read_csv('data/table_sante/basket_ML_final.csv', sep=';')

print ("2. Separating features and classes... \n")
del df['Unnamed: 0']
df.ALLERGY = df.ALLERGY.astype(int)
df_X= df.drop(['ALLERGY' ], axis=1).as_matrix().astype(int)
df_y = df.ALLERGY

#### Divide Training Data and Test Data 80%:20%
#### Using train_test_split function, for X, should be in the type of 'numpy.ndarray', y 'pandas.core.series.Series'

print ("3. Separating training data and test data... \n")
df_X_train, df_X_test, df_y_train, df_x_test = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

#### this method randomly divides df into traing and test data with the proprotion 80%:20% 
#### ex: sum(df_y_train == 1) / sum(df_y==1) == 80%

print ("4. PCA transformation... \n") 
pca = PCA(n_components=40)
df_X_train_pca = pca.fit_transform(df_X_train)
df_X_test_pca = pca.fit_transform(df_X_test)

print ("4.1 Save PAC transformed training and test data... \n") 
os.path.join('data/table_sante/', 'pca_x_train')
np.savetxt('data/table_sante/pca_x_train', df_X_train_pca)

os.path.join('data/table_sante/', 'pca_x_test')
np.savetxt('data/table_sante/pca_x_test', df_X_test_pca)
"""

def writeResult(model):
	print ("7. Writing summary: ")
	os.path.join('data/table_sante/', 'svm_summary_allergy.txt')
	file = open ('data/table_sante/svm_summary_allergy.txt', 'a')
	file.write('Summary: \n')
	file.write('-'*50)
	file.write('\n')


	file.write('All the combinations of parameters: \n')
	file.write(str(model.get_params))
	file.write('\n')
	file.write('\n')
	file.write('-'*50)
	file.write('\n')
	file.write('\n')

	file.write('All the scores in differnt combinations of parameters: \n')
	for item in model.grid_scores_:
		file.write('%s\n' %str(item))
	file.write('\n')
	file.write('-'*50)
	file.write('\n')
	file.write('\n')

	file.write('Best estimator: \n')
	file.write(str(model.best_estimator_))
	file.write('\n')
	file.write('\n')
	file.write('-'*50)
	file.write('\n')
	file.write('\n')

	file.write('Best score: \n')
	file.write(str(model.best_score_))
	file.write('\n')
	file.write('\n')
	file.write('-'*50)
	file.write('\n')
	file.write('\n')

	file.write('Best parameters: \n')
	file.write(str(model.best_params_))
	file.write('\n')
	file.write('\n')
	file.write('-'*50)
	file.write('\n')
	file.write('\n')
		
	file.write('Prediction of Class: \n')
	pred = model.predict(df_X_train_pca)
	os.path.join('data/table_sante/', 'svm_best_pred_allergy.txt')
	np.savetxt('data/table_sante/svm_best_pred_allergy.txt', pred)
	file.write(str(pred))
	file.write('\n')
	file.write('\n')
	file.write('-'*50)
	file.write('\n')
	file.write('\n')

	file.write('Probability of prediction: \n')
	proba = model.predict_proba(df_X_train_pca)
	os.path.join('data/table_sante/', 'svm_best_proba_allergy.txt')
	np.savetxt('data/table_sante/svm_best_proba_allergy.txt', proba)
	file.write(str(proba))
	file.write('\n')
	file.write('\n')
	file.write('-'*50)
	file.write('\n')
	file.write('\n')

	file.close()

if __name__ == '__main__':	
	print ("1-3. Zapping... \n")
	print ("4. Reading training data... \n")
	df_X_train_pca = np.loadtxt('data/table_sante/pca_x_train')
	df_y_train = np.loadtxt('data/table_sante/y_train.txt')

	print ("5. GrideSearchCV defining... \n")

	params = [
		{'C': [10, 100], 'kernel': ['linear']},
		{'C': [10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
	]

	svc = SVC(probability=True)
	model = grid_search.GridSearchCV(svc, params, verbose=1, cv=10, n_jobs=-1)

	print ("6. Start to test different parameters... ")
	
	model.fit(df_X_train_pca, df_y_train)
	writeResult(model)
	print ("Done")