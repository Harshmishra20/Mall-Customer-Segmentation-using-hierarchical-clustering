# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 22:32:31 2023

@author: Dell
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\Data Science\Daily Practice\April\14-04-2023\3.1. AGGLOMERATIVE CLUSTERING\Mall_Customers.csv")

x=dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

df1= pd.DataFrame(x,columns=['income','spending_score'])
df1['predict_hierarchycluster']= y_hc
df1.to_csv('hc_Prediction.csv')

import os
os.getcwd()