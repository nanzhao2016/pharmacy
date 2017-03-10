from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col, lit, concat
import pyspark.sql.functions, os
from pyspark.sql.functions import col,udf, unix_timestamp
from datetime import datetime
from pyspark.sql.types import DateType
### Create sc, sqlContext
sc = SparkContext("local", "Simple APP")
sqlContext = SQLContext(sc)


tickets = sc.textFile('C:/Users/OPEN/Documents/NanZHAO/Projet CDS/data/All_data_clean.csv').persist()
header = tickets.first()
tickets = tickets.filter(lambda x: x!=header).persist()
tickets = tickets.map(lambda x: x.split(';')).persist()
header = header.split(';')
df = sqlContext.createDataFrame(tickets, header)
print("DataFrame done")

df = df.withColumn('Temps_estime', df['Temps_estime'].cast('double'))
df = df.withColumn('Temps_passe', df['Temps_passe'].cast('double'))

df = df.withColumn('Mis_a_jour', df['Mis_a_jour'].cast('timestamp'))
df = df.withColumn('Cree', df['Cree'].cast('timestamp'))


func =  udf (lambda x: datetime.strptime(x, '%Y-%m-%d'), DateType())
df = df.withColumn('Debut', func(df['Debut']))
df = df.withColumn('Sprint_date', func(df['Sprint_date']))
df = df.withColumn('Version_cible_date', func(df['Version_cible_date']))



df_tickets.coalesce(1).write.mode('append').json('C:/Users/OPEN/Documents/NanZHAO/Projet CDS/data/CDS2.json')

"""
tickets = tickets.map(lambda x : x.split(';'))

header = ["X_","Projet","Tracker","Statut","Priorité","Sujet","Auteur","Assigné_à","Mis_à_jour","Version_cible","Début","Temps_estimé",                
	"Temps_passé","X_réalisé","Créé","Demandes_liées","Sprint","MOA","AMOA","Nature","En_attente","A_Développer_Sprint.suivant",
	"Impact.sur.le.site","Importance.métier","Coût.d.implémentation.estimé","Fonctionnalité.impactée","Description","Sprint.id",                   
    "Sprint.date","Version.cible.id","Version.cible.date"]

 """