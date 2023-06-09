

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf.fit(X)

y_pred = clf.predict(X)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Isolation Forest Outlier Detection on IRIS Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Local Outlier Factor Outlier Detection on IRIS Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# Fit Isolation Forest model
clf_iso = IsolationForest(contamination=0.1, random_state=42)
y_pred_iso = clf_iso.fit_predict(X)

# Fit Local Outlier Factor model
clf_lof = LocalOutlierFactor(contamination=0.1)
y_pred_lof = clf_lof.fit_predict(X)

# Plot Isolation Forest outliers
plt.scatter(X[:, 0], X[:, 1], c=np.where(y_pred_iso == -1, 'red', 'blue'), label='Isolation Forest')
plt.title("Outlier Detection using Isolation Forest on IRIS Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

# Plot Local Outlier Factor outliers
plt.scatter(X[:, 0], X[:, 1], c=np.where(y_pred_lof == -1, 'red', 'blue'), label='Local Outlier Factor')
plt.title("Outlier Detection using Local Outlier Factor on IRIS Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

