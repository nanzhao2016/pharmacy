from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col, lit, concat, udf
from pyspark.sql.types import *
from datetime import datetime
import pyspark.sql.functions, os, time 


### Create sc, sqlContext
sc = SparkContext("local", "Simple APP")
sqlContext = SQLContext(sc)


### Load patients, pharmas, produits from saved parquet 
all = sqlContext.read.parquet('data/table_sante/tableAll_org')


all = all.drop("famille_niv2").drop("famille_niv3").drop("famille_niv4")
all = all.withColumn("location", concat(col("latitude"), lit(","), col("longitude")))


#### Method 1 change date format to yyyy-mm-dd
def date_str (string):
	return (string[:4]+"-"+string[4:6]+"-"+string[6:8])
udf_date_str = udf(date_str, StringType())
all = all.withColumn("date", udf_date_str("ev_id_date"))
all = all.withColumn("date", all.date.cast(DateType()))

#### Method 2 change date format to yyyy-mm-dd datetime.strptime()
def toDate (string):
	return datetime.strptime(string, "%Y%m%d")
udf_toDate = udf(toDate, DateType())
all = all.withColumn("date2", udf_toDate("ev_id_date"))

#### Change data of DateType to TimestampType: yyyy-mm-dd to yyyy-mm-dd HH:MM:SS 
all = all.withColumn("date_stamp", all.date2.cast(TimestampType()))

#### Change data of StringType to TimestampType: yyyymmddHHMM to yyyy-mm-dd HH:MM:SS datetime ()
def toDate2 (string):
	return datetime(int(string[:4]), int(string[4:6]), int(string[6:8]), int(string[8:10]), int(string[10:12]))
udf_toDate2 = udf(toDate2, TimestampType())
all = all.withColumn("date_stamp2", udf_toDate2("lv_date_vente"))

### Change data of DateType (yyyy-mm-dd) to a long integer DoubleType : time.mktime, date.timetuple
def toTimestamp(date):
	return time.mktime(date.timetuple())
udf_toTimestamp = udf(toTimestamp, DoubleType())
all = all.withColumn("timestamp", udf_toTimestamp("date2"))
all = all.withColumn("timestamp2", udf_toTimestamp("date_stamp2"))

### Another method to change date to a long integer 
def changeToTimeStamp(day):
	return time.mktime(datetime(day[:4], day[4:6], day[6:8]).timetuple())
	
#print (len(all.columns))
#print (all.printSchema())
#print(all.show(1))

### Change columns type
double_list=['ev_total_remise_facture', 'ev_total_facture', 'ev_age', 'lv_quantite', 'lv_prix', 'lv_promotion_remise', 'lv_total_remise', 'lv_id_tva', 'population', 'latitude', 'longitude']

for col in double_list:
	all = all.withColumn(col, all[col].cast(DoubleType()))
	
#print (len(all.columns))
#print (all.printSchema())
#print(all.show(1))

all.write.parquet('data/table_sante/tableAll_withTime')

all_json = all.drop("ev_id_date").drop("lv_date_vente").drop("latitude").drop("longitude").drop("date2")
#print (all_json.columns)
#print (all_json.printSchema())

all_json.write.parquet('data/table_sante/tableAll_prepareto_json')
all_json.write.json('data/table_sante/tickets.json')



