from pyspark import SparkContext, SparkConf, SQLContext, RDD
from pyspark.sql.functions import col, lit, concat
import pyspark.sql.functions, os


### Create sc, sqlContext
sc = SparkContext("local", "Simple APP")
sqlContext = SQLContext(sc)


### Load data by textFile
patients = sc.textFile('C:/Users/OPEN/Documents/Data/Openhealth/Patient/res2.csv')
pharmas = sc.textFile('C:/Users/OPEN/Documents/Data/Openhealth/Patient/pharmas_milieu_urbain.csv')
produits = sc.textFile('C:/Users/OPEN/Documents/Data/Openhealth/Patient/produits_uniques_clean.csv')


### Change to dataframe: patients
patients = patients.map(lambda x : x.split(';'))
header = ["ev_id_facture", "ev_id_pharmacie", "ev_id_client", "ev_flag_vignette_avancee",  "ev_total_remise_facture", "ev_total_facture", "ev_id_vendeur", "ev_id_date", "ev_id_type_facture", "ev_num_renouvelement", "ev_id_type_prescription", "ev_flag_facture_annule" ,  "ev_id_facture_annule", "ev_id_medecin", "ev_type_maj", "ev_age", "ev_sexe", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "lv_id_lventes", "lv_id_pharmacie", "lv_id_facture","lv_id_produit", "lv_quantite", "lv_prix", "lv_promotion_remise", "lv_total_remise", "lv_type_conditionnement", "lv_motant_rembourse_ro", "lv_motant_rembourse_rc", "lv_id_substitution", "lv_id_act", "lv_id_tva", "lv_id_non_substituable", "lv_id_vendeur", "lv_id_medecin", "lv_date_vente", "lv_type_maj", "V9", "V10", "V11", "V12", "V13", "V14", "V15"]
df_patients = sqlContext.createDataFrame(patients, header) 
#print (df_patients.printSchema())
#print (df_patients.take(1))
#print (df_patients.count())
#print (df_patients.distinct().count())


### Change to dataframe: pharmas 
header = pharmas.first()
pharmas = pharmas.filter(lambda x: x!=header)
pharmas = pharmas.map(lambda x: x.split(";"))
header = header.split(";")
df_pharmas = sqlContext.createDataFrame(pharmas, header)
#print (df_pharmas.printSchema())


### Change to dataframe: produits
header = produits.first()
produits = produits.filter(lambda x: x!=header)
produits = produits.map(lambda x: x.split(";"))
header = header.split(";")
	
#print (df_produits.printSchema())


### First time to write into parquet: patients, pharmas and produits  
#df_patients.write.parquet('data/table_sante/patients_org')
#df_pharmas.write.parquet('data/table_sante/pharmas')
#df_produits.write.parquet('data/table_sante/produits')

### Overwrite into parquet: patients, pharmas and produits 
df_patients.write.mode('overwrite').parquet(os.path.join('data/table_sante', 'patients_org'))
df_pharmas.write.mode('overwrite').parquet(os.path.join('data/table_sante', 'pharmas'))
df_produits.write.mode('overwrite').parquet(os.path.join('data/table_sante', 'produits'))


### Remove inutil columns from patients 
drop_col =['ev_flag_vignette_avancee', 'ev_id_vendeur', 'ev_id_type_facture', 'ev_num_renouvelement', 'ev_flag_facture_annule', 'ev_flag_facture_annule', 'ev_id_facture_annule', 'ev_id_medecin', 'ev_type_maj', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'lv_id_pharmacie', 'lv_id_facture', 'lv_type_conditionnement', 'lv_motant_rembourse_ro', 'lv_motant_rembourse_rc', 'lv_id_substitution', 'lv_id_act', 'lv_id_non_substituable', 'lv_id_vendeur', 'lv_id_medecin', 'lv_type_maj', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15']
df_patients_sub = df_patients.select([column for column in df_patients.columns if column not in drop_col])
#print (df_patients_sub.printSchema())
#print (df_patients_sub.take(1))
#print (df_patients_sub.count())
#print (df_patients_sub.distinct().count())


### Remove duplicated lines in the dataframe  
df_patients_sub = df_patients_sub.distinct()
#print (df_patients_sub.count())


### First time to write into parquet: patients_clean 
#df_patients_sub.write.parquet('data/table_sante/patients_clean')

### Overwrite into parquet: patients_clean 
df_patients_sub.write.mode('overwrite').parquet(os.path.join('data/table_sante','patients_clean'))


### Jointure 
all = df_patients_sub.join(df_pharmas, df_patients_sub['ev_id_pharmacie'] == df_pharmas['ev_id_pharmacie'], "left").drop(df_pharmas['ev_id_pharmacie'])
all = all.join(df_produits, all['lv_id_produit'] == df_produits['lv_id_produit'], "left").drop(df_produits['lv_id_produit'])
#print (all.printSchema())
#print (all.count())

### First time to write into parquet: tableAll  
all.write.parquet('data/table_sante/tableAll_org')

### Overwrite into parquet: tableAll
#all.write.mode('overwrite').parquet(os.path.join('data/table_sante', 'tableAll_org'))







