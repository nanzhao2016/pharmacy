from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col
import pyspark.sql.functions
import os, json, pprint
from collections import defaultdict


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

def mapReduceBasket(list_org, list_red):
	for item in list_org:
		new_dict = {}
		d = defaultdict(int)
		for k,v in zip(item['shortName'], item['lv_quantite']):
			d[k] += v
		new_dict['basket'] = dict(d)
		new_dict['ev_id_facture']=item['ev_id_facture']
		list_red.append(new_dict)

file = open(os.path.join('data/table_sante/', 'dict_filted.csv'))
list_org = []
readFromCsv(list_org, file)
print ("Original list reading done, length: ", len(list_org))
print ("-"*50)
file.close()

list_red = []
mapReduceBasket(list_org, list_red)
print ("Mapreduced list reading done, length: ", len(list_red))
print ("-"*50)

file = open(os.path.join('data/table_sante/', 'dict_MR.csv'), 'a')
writeToCsv(list_red, file)
print ("Writing done")
file.close()




