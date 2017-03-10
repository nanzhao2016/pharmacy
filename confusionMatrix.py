import pandas as pd
import numpy as np
import pprint, os

from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

#os.path.join('data/table_sante/','confusion_table_KNN.png')

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, names = ['Non_Allergy', 'Allergy']):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	#plt.savefig('data/table_sante/confusion_table_KNN.png')
	

	
def descibe_confusion_matrix(y_true, y_pred, algo, file):
	file.write("Confusion table of ") 
	file.write(algo)
	file.write("\n")
	
	m_accuracy = metrics.accuracy_score(y_true, y_pred)
	file.write("Accuracy: ") 
	file.write(str(m_accuracy))
	file.write("\n")
	
	mcc = metrics.matthews_corrcoef(y_true, y_pred)
	file.write("MCC: ")
	file.write(str(mcc))
	file.write("\n")
	
	names = ['Non_Allergy', 'Allergy']
	report = metrics.classification_report(y_true, y_pred, target_names = names)
	file.write("Classification report: \n") 
	file.write(str(report))
	file.write("\n")
	
	matrix = metrics.confusion_matrix(y_true, y_pred)
	file.write("Confusion matrix, without normalization: \n") 
	file.write(str(matrix))
	file.write("\n")
	
	# Normalize the confusion matrix by row (i.e by the number of samples in each class)
	matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
	file.write("Normalized confusion matrix: \n") 
	file.write(str(matrix_normalized))
	file.write("\n")
	
	file.write("-"*50)
	file.write("\n")
	
	return matrix_normalized
	
def save_confusion_matrix(matrix_normalized, algo):
	plt.figure()
	plot_confusion_matrix(matrix_normalized, title='Normalized confusion matrix '+algo)
	
	algo = algo.replace(" ", "")
	os.path.join('data/table_sante/','confusion_table_'+algo+'.png')
	fig = plt.gcf()
	fig.set_size_inches(8, 8, forward=True)
	fig.savefig('data/table_sante/confusion_table_'+algo+'.png')
	#plt.show()
	
def save_predicted(clf, X_test, y_test, algo):
	#predicted = cross_val_predict(clf, X_test, y_test, cv=20, n_jobs=-1)
	print(algo)
	print ("\n")
	clf.fit(X_test, y_test)
	print(clf)
	print ("\n")
	predicted = clf.predict(X_test)
	algo = algo.replace(" ", "")
	os.path.join('data/table_sante/', 'y_test_'+algo+'.txt')
	np.savetxt('data/table_sante/y_test_'+algo+'.txt', predicted)
	
"""	
X_train = np.loadtxt('data/table_sante/pca_x_train')
y_train = np.loadtxt('data/table_sante/y_train.txt')
"""


X_test = np.loadtxt('data/table_sante/pca_x_test')
y_test = np.loadtxt('data/table_sante/y_test.txt')


y_KNN = np.loadtxt('data/table_sante/y_test_KNN.txt')
y_RF = np.loadtxt('data/table_sante/y_test_RF.txt')
y_LDA = np.loadtxt('data/table_sante/y_test_LDA.txt')
y_Extra = np.loadtxt('data/table_sante/y_test_ExtraTrees.txt')


"""
if __name__ == '__main__':	
	
	
	print("RF ...\n")
	algo = "RF"
	clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_samples_leaf=5, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=True, random_state=None, verbose=1, warm_start=False)
	save_predicted(clf, X_test, y_test, algo)
	print("Done \n")
	
	print("KNN ...\n")
	algo = "KNN"
	clf = KNC(algorithm='auto', leaf_size=30, metric='manhattan',
           metric_params=None, n_jobs=-1, n_neighbors=100, p=2,
           weights='distance')
	save_predicted(clf, X_test, y_test, algo)
	print("Done \n")

	print("LDA ...\n")
	algo = "LDA"
	clf = LDA(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)
	save_predicted(clf, X_test, y_test, algo)
	
	
	print("Extra Trees ... \n")
	algo = "Extra Trees"
	clf = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features=None, max_leaf_nodes=None,
           min_samples_leaf=3, min_samples_split=10,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
	save_predicted(clf, X_test, y_test, algo)
	print("Done \n")
"""	




#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(matrix, title='Confusion matrix')

os.path.join('data/table_sante/','info_confusion_table.txt')
file = open('data/table_sante/info_confusion_table.txt', 'a')


algo = "Nearest Neighbors"
matrix_normalized = descibe_confusion_matrix(y_test, y_KNN, algo, file)
save_confusion_matrix(matrix_normalized, algo)

algo = "Random Forest"
matrix_normalized = descibe_confusion_matrix(y_test, y_RF, algo, file)
save_confusion_matrix(matrix_normalized, algo)

algo = "Linear Discriminant Analysis"
matrix_normalized = descibe_confusion_matrix(y_test, y_LDA, algo, file)
save_confusion_matrix(matrix_normalized, algo)

algo = "Extra Trees"
matrix_normalized = descibe_confusion_matrix(y_test, y_Extra, algo, file)
save_confusion_matrix(matrix_normalized, algo)
file.close()

