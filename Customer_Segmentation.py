import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data=pd.read_csv('Mall_Customers.csv')

# # show 5 rows
# print(customer_data.head())

# # show row, colum numbers
# print(customer_data.shape)

# # getting info about data
# print(customer_data.info)

# # check for missing values
# print(customer_data.isnull().sum())

# choosing the Annual income and spendidng column
X=customer_data.iloc[:,[3,4]].values
# print(X)

# choosing the no of cluster
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plot an elbow graph
sns.set()
plt.plot(range(1,11), wcss)
plt.title('The elbow point Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# plt.show()

# optimum no of clusters=5

# training the k_means clustering model
kmeans=KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y=kmeans.fit_predict(X)
print(Y)

# visualizing all the clusters
# plotting all the clusters and their centroid
# cluters - 0,1,2,3,4
plt.figure(figsize=(8,8))
plt.scatter(X[Y== 0,0], X[Y== 0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y== 1,0], X[Y== 1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y== 2,0], X[Y== 2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y== 3,0], X[Y== 3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y== 4,0], X[Y== 4,1], s=50, c='blue', label='Cluster 5')

# plot the centroid
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()