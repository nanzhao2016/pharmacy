from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.types import *
import json, os

sc = SparkContext("local", "Simple APP")
sqlContext = SQLContext(sc)

def createPath(path):
	if not os.path.exists(path):
		os.makedirs(path)	
	#print 'path done'

	
def writeToJson(df, f, c):
	for row in df.rdd.collect():
		index_dict =  {"index":{"_id":str(c)}}
		row_dict = row.asDict()
		f.write(str(json.dumps(index_dict)))
		f.write('\n')
		f.write(str(json.dumps(row_dict)))
		f.write('\n')
		c += 1
	return (c)

def addJsonFile(prefix, suffix, c, N):
	for i in N:
		filepath = os.path.join(path, str(i)+'.json')
		f = open(filepath, 'a')
		df = sqlContext.read.json(prefix+str(i)+suffix)
		c = writeToJson(df, f, c)
		f.close()
	return c 
	
path = 'data/Indices/openhealth_v2' 
createPath(path)
c = 1
suffix = '-dd6ecce6-3713-4ada-b17b-384e403a2462'
N = range(0,10)
prefix = 'data/table_sante/tickets_v2.json/part-r-0000'
c = addJsonFile(prefix, suffix, c, N)

N = range(10, 100)
prefix = 'data/table_sante/tickets_v2.json/part-r-000'
c = addJsonFile(prefix, suffix, c, N)

N = range(100, 200)
prefix = 'data/table_sante/tickets_v2.json/part-r-00'
c = addJsonFile(prefix, suffix, c, N)



