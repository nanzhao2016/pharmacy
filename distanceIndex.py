import pandas as pd
import numpy as np 
import math
import os

df = pd.read_csv('C:/Users/OPEN/Documents/Data/Openhealth/Patient/matrix/oralair_purchase_index_pure.csv', sep = ',', header=0)
df.id_client_unique = df.id_client_unique.astype('str')
ids = df.id_client_unique.unique()

df_100 =  df[['id_client_unique', 'ORALAIR_100IR']]
df_300 =  df[['id_client_unique', 'ORALAIR_300IR']]


d = pd.DataFrame(0, index=np.arange(ids.size), columns=np.arange(ids.size))

print ("Start to calculate the matrix... \n")
for i in range(ids.size):
	for j in range(ids.size):
		ids_300_i = df_300.loc[df_300['id_client_unique']==ids[i]]
		ids_300_j = df_300.loc[df_300['id_client_unique']==ids[j]]
		ids_100_i = df_100.loc[df_100['id_client_unique']==ids[i]]
		ids_100_j = df_100.loc[df_100['id_client_unique']==ids[i]]
		d[i][j] = math.sqrt(sum((ids_300_i['ORALAIR_300IR']-ids_300_j['ORALAIR_300IR'])**2 + (ids_100_i['ORALAIR_100IR']-ids_100_j['ORALAIR_100IR'])**2))
		print ((i,j))

d = d.fillna(0)
os.path.join('data/table_sante/', 'distance.txt')
np.savetxt('data/table_sante/distance.txt', d)

print ("Done")
