from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col, lit, concat, udf, split
from pyspark.sql.types import *
from datetime import datetime
import pyspark.sql.functions, os, time
from pyspark.sql import Column, Row

### Create sc, sqlContext
sc = SparkContext("local", "Simple APP")
sqlContext = SQLContext(sc)

df = sqlContext.read.json('data/table_sante/tickets.json')
df_sub = df.drop("date_stamp").drop("date_stamp2").drop("timestamp").drop("timestamp2")
df_subNotNULL = df_sub.where(col("name").isNotNull())
print (df_subNotNULL.count())

def shortenName (name):
	if name == "A 313 POM TUB 50G":
		return "A 313" 
	elif name.startswith("L107 LEHNING") or name.startswith("L114 LEHNING") or name.startswith("L25 LEHNING") or name.startswith("L28 LEHNING") or name.startswith("L52 LEHNING") or name.startswith("L72 LEHNING") or name.startswith("L8 LEHNING"):
		return "LEHNING GTT"
	elif name.startswith("ONE TOUCH"):
		return "ONE TOUCH"
	elif name == "OPO VEINOGENE SOL BUV 150ML":
		return "OPO VEINOGENE"
	elif name == "PO 12 2% CR TUB 40G":
		return "PO 12"
	elif name.startswith("RHUS TOX"):
		return "RHUS TOX"
	elif name == "TARKA LP CPR 28":
		return "TARKA LP"
	elif name.startswith("TRAMADOL/PARAC"):
		return "TRAMADOL"
	elif name.startswith("ULTRA LEVURE"):
		return "ULTRA LEVURE"
	elif name.startswith("ULTRALEVURE"):
		return "ULTRA LEVURE"
	else:
		return name.split(" ")[0]
		
udf_shortenName = udf(shortenName, StringType())

#print (df_subNotNULL.withColumn("shortName", udf_shortenName(col("name"))).show(100))
df_subNotNULL = df_subNotNULL.withColumn("shortName", udf_shortenName(col("name")))
	
#print (df_sub.printSchema())
#print (df_subNotNULL.select("shortName").distinct().orderBy("shortName").show(100))

#print (df_sub.select("name").where(col("name").startswith("L")).distinct().orderBy("name").show(50))

df_to_dict = df_subNotNULL
df_to_dict.write.json('data/table_sante/tickets_dic.json')

#tickets_org = []
#for row in df_to_dict.rdd.collect():
#	row_dict=row.asDict()
#	tickets.append(row_dict)

#print (len(tickets_org))