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
plt.show()

# optimum no of clusters=5

# training the k_means clustering model
kmeans=KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y=kmeans.fit_predict(X)
print(Y)

# visualizing all the clusters
