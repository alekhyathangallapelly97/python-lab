
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

white_wine_data = pd.read_csv('./dataset/winequality_white.csv')

# Null values
null_values = pd.DataFrame(white_wine_data.isnull().sum().sort_values(ascending=False)[:25])
null_values.columns = ['Null Count']
null_values.index.name = 'Feature'
print(null_values)

# handling the missing value
final_data = white_wine_data.select_dtypes(include=[np.number]).interpolate().dropna()

# find the most correlated features
numeric_data = white_wine_data.select_dtypes(include=[np.number])
corrl = numeric_data.corr()
print (corrl['quality'].sort_values(ascending=False)[:4], '\n')

# Preprocessing the data
scaler_data = preprocessing.StandardScaler()
scaler_data.fit(final_data)
X_scaled_array = scaler_data.transform(final_data)
X_scaled_final = pd.DataFrame(X_scaled_array, columns = final_data.columns)

wcss = []
# elbow method to know the number of clusters
for i in range(2,12):
    kmeans_model = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans_model.fit(final_data)
    wcss.append(kmeans_model.inertia_)
    silh_score = silhouette_score(final_data, kmeans_model.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, silh_score))

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()