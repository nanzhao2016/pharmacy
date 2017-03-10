from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

"""
df = pd.read_csv('data/table_sante/basket_pandas_df.csv', sep=';')
number_0 = sum((df == 0).sum(axis=1))
print ("Content equals to 0: ", number_0)
print ("Matrix size: ", df.size)
"""
"""
#### METHOD 1 TO ADD A NEW COLUMN
#### generate a column allerge: for the ticket with oralair or grazax, allergy is true, otherwise, allergy is false  
oralair = df['ORALAIR']
grazax = df['GRAZAX']
allergy = pd.DataFrame(oralair + grazax)
allergy.columns = ['ALLERGY']
allergy = (allergy != 0)

#### join two dataframe togther 
df_ML = pd.concat([df, allergy], axis=1)

print ("First method to show line 3")
print ("df_ML.as_matrix()[2]")
print ("-"*50)
print (df_ML.as_matrix()[2])

print ("-"*50)
print ("Seconde method to show line 3")
print ("df_ML[2:3]")
print (df_ML[2:3])

print ("-"*50)
print ("Thirs method to show line 3")
print ("df_ML.loc[[2]]")
print (df_ML.loc[[2]])

print ("-"*50)
print ("Fourth methos to show line 3")
print ("df_ML.ix[2]")
print (df_ML.ix[2])
"""

"""
#### METHOD 2 TO ADD A NEW COLUMN
df['ALLERGY'] = df['ORALAIR']+df['GRAZAX']
df['ALLERGY'] = (df['ALLERGY'] !=0)
#print (df[0:2])
del df['Unnamed: 0']
del df['ORALAIR']
del df['GRAZAX']
#print (df[0:2])
df.ALLERGY = df.ALLERGY.apply(str)

df.to_csv('data/table_sante/basket_ML.csv', sep=';', header=True, encoding='utf-8')
"""
#### Compressed Sparse Row Format ####
## indices, indptr, data 
## indices: is array of column indices 
## data: is array of corresponding nonzero values
## indptr: points to row starts in indices and data 
#import scipy.sparse as sparse
#df_dense = df.as_matrix()

def getZeroRow(df_ML_x, list_index):
	for i in range(len(df_ML_x)):
		if (((df_ML_x.loc[[i]]==0).sum(axis=1))==2382).bool():
			list_index.append(i)

df = pd.read_csv('data/table_sante/basket_ML.csv', sep=';')
df_ML = df.drop('Unnamed: 0',1)
df_ML_x = df_ML.drop('ALLERGY',1)
df_ML_y = df[['ALLERGY']]

list_index=[]
getZeroRow(df_ML_x, list_index)

df_ML_final = df_ML.drop(df_ML.index[list_index])
print ("df_ML shape: ", df_ML.shape)
print ("df_ML_final shape: ", df_ML_final.shape)
number_0 = sum((df_ML_final == 0).sum(axis=1))
print ("df_ML_final content equals to zero: ", number_0)
print ("df_ML_final content size: ", df_ML_final.size)
df_ML_final.to_csv('data/table_sante/basket_ML_final.csv', sep=';', header=True, encoding='utf-8')
print ("Writing done")
#### get all columns names : df.columns.values.tolist() 