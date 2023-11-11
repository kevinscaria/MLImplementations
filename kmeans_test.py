import pandas as pd
from ucimlrepo import fetch_ucirepo 

from sklite.clustering import KMeans
from sklite.metrics import accuracy_score
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
X = iris.data.features 
y = iris.data.targets['class'].map({"Iris-versicolor":0, "Iris-setosa":1, "Iris-virginica":2}).values

kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(X_train=X)
preds = kmeans.predict(X)
print("Accuracy Score: ", round(accuracy_score(preds, y), 3))