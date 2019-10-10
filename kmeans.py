import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import time
import matplotlib.pyplot as plt
import warnings


import pandas as pd
import numpy as np

import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
warnings.filterwarnings('ignore')

# some function for later
outliers_fraction = 0.01

# return Series of distance between each point and his distance with the closest centroid
def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance


df = pd.read_excel("Signal Detection.xlsx")

data= df.copy()
df_siteid = data['Site ID']
df_siteid=pd.DataFrame(df_siteid)

data = data[['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']]
print(data.info())

min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# reduce to 2 importants features
pca = PCA(n_components=2)
data = pca.fit_transform(data)
# standardize these 2 new features
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)

# calculate with different number of centroids to see the loss plot (elbow method)
n_cluster = range(1, 30)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()

# Not clear for me, I choose 15 centroids arbitrarily and add these data to the central dataframe
df['cluster'] = kmeans[25].predict(data)
df['principal_feature1'] = data[0]
df['principal_feature2'] = data[1]
df['cluster'].value_counts()
print(df['principal_feature1'])
print(df['principal_feature2'])
print("==================================")
print(df['cluster'].value_counts())

fig, ax = plt.subplots()
colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown', 9:'purple', 10:'white', 11: 'grey', 12:'lightblue', 13:'lightgreen', 14: 'darkgrey',15:'yellow',16:'black',17:'brown',18:'white',19:'pink',20:'red',21:'magenta',22:'blue',23:'green',24:'cyan',25:'purple'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["cluster"].apply(lambda x: colors[x]))
plt.show()

# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
distance = getDistanceByPoint(data, kmeans[25])
number_of_outliers = int(outliers_fraction*len(distance))
threshold = distance.nlargest(number_of_outliers).min()
# anomaly21 contain the anomaly result of method 2.1 Cluster (0:normal, 1:anomaly)
df['anomaly21'] = (distance >= threshold).astype(int)


#for i in range(len(df['anomaly21'])):
#    if i == 1:
#        print("================")
#print(df['anomaly21'])



# visualisation of anomaly with cluster view
fig, ax = plt.subplots()
colors = {0:'blue', 1:'red'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["anomaly21"].apply(lambda x: colors[x]))
plt.show()


# y_pred=df['anomaly21']
# y_pred=pd.DataFrame(y_pred)
# print(y_pred)
# if y_pred==1:
#     print("=========")
# #
# b=[]
# for i in range(len(y_pred)):
#     if y_pred[i] == 1:
#         print(y_pred)
#print(b)

# fig, ax = plt.subplots()
#
# a = df.loc[df['anomaly21'] == 1, ['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']] #anomaly
#
# ax.plot(df['SF Ratio'], df['PD Ratio'],df['AE Ratio'], df['SAE Ratio'],df['discontinued patients Ratio'], color='blue')
# ax.scatter(a['SF Ratio'],a['PD Ratio'],a['AE Ratio'],a['SAE Ratio'], a['discon
set_siteid = []
data=data.join(df_siteid)
#print(data.info)
y_pred = [1 if p > 0.5 else 0 for p in df.anomaly21.values]
X_test_siteid = df_siteid['Site ID'].tolist()
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        set_siteid.append(X_test_siteid[i])

print('Sites with anomaly: ', set_siteid)
