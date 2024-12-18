import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data=pd.read_csv('Mall_Customers.csv')

# show 5 rows
print(customer_data.head())

# show row, colum numbers
print(customer_data.shape)

# getting info about data
print(customer_data.info)

# check for missing values
print(customer_data.isnull().sum())

# choosing the Annual income and spendidng column

X=customer_data.iloc[:,[3,4]].values
# print(X)

# choosing the no of cluster



