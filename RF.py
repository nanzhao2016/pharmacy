import pandas as pd
import numpy as np
import pprint

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import grid_search


#### First test with random subsampling - cross validation
#### Then try to use kfold

print ("Reading data... ")
df = pd.read_csv('data/table_sante/basket_ML_final.csv', sep=';')

print ("Separating features and classes... ")
del df['Unnamed: 0']
df.ALLERGY = df.ALLERGY.astype(int)
df_X= df.drop(['ALLERGY' ], axis=1).as_matrix().astype(int)
df_y = df.ALLERGY

#### Divide Training Data and Test Data 80%:20%
#### Using train_test_split function, for X, should be in the type of 'numpy.ndarray', y 'pandas.core.series.Series'

print ("Separating training data and test data... ")
df_X_train, df_X_test, df_y_train, df_x_test = train_test_split(df_X, df_y, test_size=0.2, random_state=123)

#### this method randomly divides df into traing and test data with the proprotion 80%:20% 
#### ex: sum(df_y_train == 1) / sum(df_y==1) == 80%

print ("PCA transformation... ") 
pca = PCA(n_components=40)
df_X_train_pca = pca.fit_transform(df_X_train)
df_X_test_pca = pca.fit_transform(df_X_test)

print ("GrideSearchCV defining... ")
params = {'n_estimators':[5,10,30,100], 'min_samples_leaf':[3, 5, 10, 15], 'min_samples_split':[2, 3, 5]}
rf = RandomForestClassifier(max_features=None, oob_score=True, n_jobs=-1, verbose=1)
model = grid_search.GridSearchCV(rf, params, cv=20)

print ("Start to test different parameters... ")
model.fit(df_X_train_pca, df_y_train)

print ("-"*50)
print ("Summary: ")
print ("All the combinations of parameters: ")
print (model.get_params)
print ("All the scores in differnt combinations of parameters: ")
print (model.grid_scores_)
print ("Best estimator: ")
print (model.best_estimator_)
print ("Best score: ")
print (model.best_score_)
print ("Best parameters: ")
print (model.best_params_)
print ("Prediction of Class: ")
print (model.predict(df_X_train_pca))
print ("Probability of prediction: ")
print (model.predict_proba(df_X_train_pca))


 


