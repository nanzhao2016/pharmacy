import pandas as pd
import numpy as np
import pprint

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA 

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

print ("PCA calculating... ") 
pca = PCA()
pca.fit(df_X_train)

print ("Saving components as 3, 5, 10, 20, 40, 80... ")
list_pca=[]
for i in [3, 5, 10, 20, 40, 80]:
	percentage = sum(pca.explained_variance_ratio_[:i])
	list_pca.append({i:percentage})

print ("Testing components with random forest... ")
print ("-"*50)
list_res=[]
list_res_ave=[]	
for i in [3, 5, 10, 20, 40, 80]:
	pca_i = PCA(n_components = i)
	print ("Components ", i)
	df_X_train_pca = pca_i.fit_transform(df_X_train)
	rf = RandomForestClassifier(n_estimators=10, max_features=None, min_samples_leaf=100, min_samples_split=5, oob_score=True, n_jobs=-1, verbose=1)
	scores_i = cross_val_score(rf, df_X_train_pca, df_y_train, cv=20)
	list_res.append({i:scores_i})
	list_res_ave.append({i:scores_i.mean()})
	print ("-"*50)
	print ("-"*50)


print ("Percentage of explanation PCA: ")
pprint.pprint(list_pca)	
print ("-"*50)
print ("Accuracy of different PCA: ")
pprint.pprint(list_res)
pprint.pprint(list_res_ave)	

"""
Percentage of explanation PCA:
[{3: 0.65725559647092657},
 {5: 0.75895732130971727},
 {10: 0.82269753137553481},
 {20: 0.88883112492282323},
 {40: 0.93581167895452155},
 {80: 0.96784429076653855}]
--------------------------------------------------
Accuracy of different PCA:
[{3: array([ 0.95177665,  0.95177665,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523])},
 {5: array([ 0.95177665,  0.95177665,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523])},
 {10: array([ 0.95177665,  0.95177665,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523])},
 {20: array([ 0.95177665,  0.95177665,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523])},
 {40: array([ 0.95177665,  0.95177665,  0.95221607,  0.95198523,  0.95198523,
        0.95198523,  0.95221607,  0.95221607,  0.95198523,  0.95198523,
        0.95198523,  0.95221607,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95221607])},
 {80: array([ 0.95177665,  0.95177665,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95198523,  0.95198523,  0.95198523,  0.95198523,
        0.95198523,  0.95175439,  0.95198523,  0.95198523,  0.95198523])}]
[{3: 0.9519643685757273},
 {5: 0.9519643685757273},
 {10: 0.9519643685757273},
 {20: 0.9519643685757273},
 {40: 0.9520220786403627},
 {80: 0.95195282656280023}] 
"""
"""
We choose 40 as the reduced components 
"""


