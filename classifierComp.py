import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn import datasets



names = ["Nearest Neighbors", "Random Forest", "Linear Discriminant Analysis"]

classifiers = [
    KNC(algorithm='auto', leaf_size=30, metric='manhattan', metric_params=None, n_neighbors=100, p=2, weights='distance', n_jobs=-1),
    RandomForestClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=5, max_features=None, oob_score=True, n_jobs=-1),
    LDA(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
]


"""
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
"""
h = .1  # step size in the mesh
print("Loading data")
X = np.loadtxt('data/table_sante/pca_x_test')[:, :2]
y = np.loadtxt('data/table_sante/y_test.txt')

"""
y_KNN = np.loadtxt('data/table_sante/y_test_KNN.txt')
y_RF = np.loadtxt('data/table_sante/y_test_RF.txt')
y_LDA = np.loadtxt('data/table_sante/y_test_LDA.txt')
"""

# Create color maps

"""
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
"""
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].



"""
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
"""

if __name__ == '__main__':	
	for name, clf in zip(names, classifiers):
		print(name)
		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		print("Fiting and scoring")
		clf.fit(X, y)
		score_ = clf.score(X, y)
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)

		plt.figure()
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
	
		print("Plot also the training points")
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title(name)
		plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score_).lstrip('0'), size=15, horizontalalignment='right')

		print("Plot lengend")
		#pointtypes = iris.target_names
		pointtypes = np.array(['Non_Allergy', 'Allergy'])
		#32proxy_artists = [patches.Circle((0, 0), 0.2, color=cmap_bold(i), label=pointtypes[i]) for i in range(3)]
		proxy_artists = [mpatches.Circle((0,0), radius = 0.2, color=cmap_bold(i)) for i in range(2)]
		plt.legend(proxy_artists, pointtypes)

plt.show()