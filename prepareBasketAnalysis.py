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

##### ! Should remove the line zero before shorten Name or add a if to verify the string is not null !#####

### Create a function to get a short name for each name 
### Attention for the name with no value
def shortenName (name):
	if name is not None:
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
	else:
		return name
		
### Create udf function in order to implement with the fucntion withColumn
udf_shortenName = udf(shortenName, StringType())

#print (df_sub.withColumn("shortName", udf_shortenName(col("name"))).show(100))
df_sub = df_sub.withColumn("shortName", udf_shortenName(col("name")))
	
#print (df_sub.printSchema())
#print (df_sub.select("shortName").distinct().orderBy("shortName").show(50))
#print (df_sub.select("name").where(col("name").startswith("L")).distinct().orderBy("name").show(50))

### Write to json 
#df_sub.write.json('data/table_sante/tickets_dic_nameWithNon.json')
df_sub.write.mode('overwrite').json(os.path.join('data/table_sante', 'tickets_dic_nameWithNon.json'))
print (df_sub.count())
print("All tickets done")
print("-"*50)

### Remove the line that name value is none 
df_subNotNull = df_sub.where(col("name").isNotNull())
#df_subNotNull.write.json('data/table_sante/tickets_dic_nameWithoutNon.json')
df_subNotNull.write.mode('overwrite').json(os.path.join('data/table_sante', 'tickets_dic_nameWithoutNon.json'))
print (df_subNotNull.count())
print("Tickets with name done")
print("-"*50)

### Remove the line that quantite is smaller than 1
df_subNotNullPositive = df_subNotNull.where(col("lv_quantite")>0)
df_subNotNullPositive.write.json('data/table_sante/tickets_dic_nameWithoutNonQuanlityPositive.json')
print (df_subNotNullPositive.count())
print("Tickets with name and quantite done")

#tickets_org = []
#for row in df_to_dict.rdd.collect():
#	row_dict=row.asDict()
#	tickets.append(row_dict)

#print (len(tickets_org))