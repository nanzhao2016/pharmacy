from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col
import pyspark.sql.functions

sc = SparkContext("local", "Simple APP")
sqlContext = SQLContext(sc)


df = sqlContext.read.json('data/table_sante/tickets.json')
print ("Total rows: " + str(df.count()))

df_sub = df.drop("date_stamp").drop("date_stamp2").drop("timestamp").drop("timestamp2")

df_sub = df_sub.where(col("name").isNotNull())
print("Rows with pill name: " + str(df_sub.count()))

print ()
print ()
print ("**************************************************")

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

for i in range(0, len(letters)):
	meds_number = df_sub.select("name").where(col("name").startswith(letters[i])).distinct().orderBy("name").count()
	print ("Starts with "+ letters[i] +": " + str(meds_number))
	print ()
	print (df_sub.select("name").where(col("name").startswith(letters[i])).distinct().orderBy("name").show(meds_number))
	print ()
	print ("**************************************************")
	print ()
