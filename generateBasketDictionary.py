from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col, lit, concat, udf, split
from pyspark.sql.types import *
from datetime import datetime
import pyspark.sql.functions, os, time, json
from pyspark.sql import Column, Row
import operator, itertools
from collections import defaultdict

### Create sc, sqlContext
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

def dataFrameToDict (df, list_org):
	for row in df.rdd.collect():
		list_org.append(row.asDict())

def dictGroupBy(list_org, k, list_grouped):
	list_org = sorted(list_org, key=operator.itemgetter(k))
	for key, group in itertools.groupby(list_org, operator.itemgetter(k)):
		list_grouped.append(list(group))

def aggValues(list_grouped, list_aggregated):
	for item in list_grouped:
		d=defaultdict(list)
		for i in range(len(item)):
			for k,v in item[i].items():
				d[k].append(v)
		for k, v in d.items():
			if k !="lv_id_lventes" and k !="lv_id_produit" and k !="cip" and k != "name" and k != "shortName" and k !="famille_niv1" and k!= "lv_quantite" and k!="lv_prix":
				d[k]=list(set(v))[0]
		list_aggregated.append(d)
		
df_dict = sqlContext.read.json('data/table_sante/tickets_dic_nameWithoutNonQuanlityPositive.json')
dictionary = []
dataFrameToDict(df_dict, dictionary)
print (len(dictionary))
print("Original dictionary done")
print("-"*50)

list1 = []
dictGroupBy(dictionary, 'ev_id_facture', list1)
print(len(list1))
print("Grouped dictionary done")
print("-"*50)

list2 = []
aggValues(list1, list2)
print(len(list2))
print("Aggregated dictionary done")
print("-"*50)

file = open(os.path.join('data/table_sante/', 'dict_saved_final.csv'), 'a')
writeToCsv(list2, file)
print("Writing done")
file.close()


