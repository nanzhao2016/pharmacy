from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col
import pyspark.sql.functions
import os, json, pprint
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

sc = SparkContext("local", "Simple APP")
sqlContext = SQLContext(sc)

def writeToCsv (list, file):
	for item in list:
		file.write(json.dumps(item))
		file.write('\n')

def readFromCsv (list, file):
	while True:
		line = file.readline().rstrip()
		if not line:
			break
		else:
			list.append(json.loads(line))

def getBasketList (list_org, list_basket, key_basket):
	for item in list_org:
		list_basket.append(item[key_basket])

file = open(os.path.join('data/table_sante/', 'dict_MR.csv'))
list_org = []
readFromCsv(list_org, file)
#list_org = list_org[0:31]
print ("Original List reading done: ", len(list_org))
print ("-"*50)
file.close()

list_basket = []
getBasketList(list_org, list_basket, 'basket') 
print ("Basket List reading done: ", len(list_basket))
#pprint.pprint(list_basket)
print ("-"*50)

V = DictVectorizer(sparse=False)
array_basket = V.fit_transform(list_basket).astype(int)
names = V.get_feature_names()
df = pd.DataFrame(array_basket, columns = names) 
print ('row number: ', len(df.index))
print ('column number', len(df.columns))
df.to_csv('data/table_sante/basket_pandas_df.csv', sep=';', header=True, encoding='utf-8')
print ("Pandas dataframe writing to csv done")
print ("-"*50)

"""
df_spark = sqlContext.createDataFrame(df)
#df_spark.printSchema()
print ("Changing pandas dataframe to spark dataframe done")
print ("row number:", df_spark.count())
print ("column number:", len(df_spark.columns))
df_spark.write.parquet('data/table_sante/basket_matrix_all')
print ("Writing to parquet done")
"""

