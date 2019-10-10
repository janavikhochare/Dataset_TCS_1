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

df = pd.read_excel("Signal Detection.xlsx")
data= df.copy()

df_siteid = data['Site ID']
df_siteid=pd.DataFrame(df_siteid)

data = data[['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']]
print(data.info())

min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
# train one class SVM
model =  OneClassSVM(nu=0.95 * outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
data = pd.DataFrame(np_scaled)
model.fit(data)
# add the data to the main
data['anomaly26'] = pd.Series(model.predict(data))
data['anomaly26'] = data['anomaly26'].map( {1: 0, -1: 1} )
print(data['anomaly26'].value_counts())

#print(data.info())
set_siteid = []
data=data.join(df_siteid)
print(data.info)
y_pred = [1 if p > 0.5 else 0 for p in data.anomaly26.values]
X_test_siteid = df_siteid['Site ID'].tolist()
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        set_siteid.append(X_test_siteid[i])

print('Sites with anomaly: ', set_siteid)

