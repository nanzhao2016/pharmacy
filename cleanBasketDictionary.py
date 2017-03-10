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

def filterDict (list_org, list_filted, key_filter, key_facture):
	for item in list_org:
		dict = {k:v for (k,v) in item.items() if k in key_filter}
		dict[key_facture] = dict[key_facture][0]
		list_filted.append(dict)
		
file = open(os.path.join('data/table_sante/', 'dict_saved_final.csv'), 'r')
dict_org = []
readFromCsv(dict_org, file)
print ("length original dictionary: ", len(dict_org))
print ("Read dictionary done")
print ("-"*50)
file.close()

dict_filted = []
filterDict(dict_org, dict_filted, ('ev_id_facture', 'shortName', 'lv_quantite'), 'ev_id_facture')
print("length filter dictionary: ", len(dict_filted))
print ("Filtering dictionary done")
print ("-"*50)

file = open(os.path.join('data/table_sante/', 'dict_filted.csv'), 'a')
writeToCsv(dict_filted, file)
print("Writing done")
file.close()