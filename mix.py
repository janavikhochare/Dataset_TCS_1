import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib.font_manager
# Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn import preprocessing
# reading the big mart sales training data

data = pd.read_excel("Signal Detection.xlsx")

df_siteid = data['Site ID']
df_siteid=pd.DataFrame(df_siteid)

df = data[['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']]
print(df.info())
#df.plot.scatter('SF Ratio', 'PD Ratio')
#df.plot.show()
from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler(feature_range=(0, 1))
#df[['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']] = scaler.fit_transform(df[['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']])
#df[['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']].head()

min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(np_scaled)
df.columns=['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']
print(df.head())

X1 = df['SF Ratio'].values.reshape(-1,1)
X2 = df['PD Ratio'].values.reshape(-1,1)
X3 = df['AE Ratio'].values.reshape(-1,1)
X4 = df['SAE Ratio'].values.reshape(-1,1)
X5 = df['discontinued patients Ratio'].values.reshape(-1,1)

X = np.concatenate((X1,X2,X3,X4,X5),axis=1)

random_state = np.random.RandomState(42)
outliers_fraction = 0.05
# Define seven outlier detection tools to be compared
classifiers = {
        'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}

xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

total=[]

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))

    # copy of dataframe
    dfx = df
    dfx['outlier'] = y_pred.tolist()
    #print(dfx.head())

    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 = np.array(dfx['SF Ratio'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX2 = np.array(dfx['PD Ratio'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX3 = np.array(dfx['AE Ratio'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX4 = np.array(dfx['SAE Ratio'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX5 = np.array(dfx['discontinued patients Ratio'][dfx['outlier'] == 0]).reshape(-1, 1)
    In = np.concatenate((IX1,IX2,IX3,IX4,IX5),axis=1)

    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 = dfx['SF Ratio'][dfx['outlier'] == 1].values.reshape(-1, 1)
    OX2 = dfx['PD Ratio'][dfx['outlier'] == 1].values.reshape(-1, 1)
    OX3 = dfx['AE Ratio'][dfx['outlier'] == 1].values.reshape(-1, 1)
    OX4 = dfx['SAE Ratio'][dfx['outlier'] == 1].values.reshape(-1, 1)
    OX5 = dfx['discontinued patients Ratio'][dfx['outlier'] == 1].values.reshape(-1, 1)
    On= np.concatenate((OX1,OX2,OX3,OX4,OX5),axis=1)
    print('OUTLIERS : ', n_outliers, 'INLIERS : ', n_inliers, clf_name)

    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    print(threshold)

    # print(data.info())
    set_siteid = []
    a = set()
    # data=data.join(df_siteid)
    # print(data.info)
    y_pred = [1 if p > 0.5 else 0 for p in dfx.outlier.values]
    X_test_siteid = df_siteid['Site ID'].tolist()
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            set_siteid.append(X_test_siteid[i])
            a.add(X_test_siteid[i])
            total.append(X_test_siteid[i])
#    print("a : ",a)
    print('Sites with anomaly: ',a)

#    plt.scatter(  c='white', s=20, edgecolor='k')
#    plt.show()

print('done')
print(total)
my_dict = {i:total.count(i) for i in total}
print("=========================================================================")
print (my_dict)     #or print(my_dict) in python-3.x

    # decision function calculates the raw anomaly score for every point
    #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    #Z = Z.reshape(xx.shape)

    # # fill blue map colormap from minimum anomaly score to threshold value
    # plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)
    #
    # # draw red contour line where anomaly score is equal to thresold
    # a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
    #
    # # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    # plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    #
    # b = plt.scatter(IX1, IX2, c='white', s=20, edgecolor='k')
    #
    # c = plt.scatter(OX1, OX2, c='black', s=20, edgecolor='k')
    #
    # plt.axis('tight')
    #
    # # loc=2 is used for the top left corner
    # plt.legend(
    #     [a.collections[0], b, c],
    #     ['learned decision function', 'inliers', 'outliers'],
    #     prop=matplotlib.font_manager.FontProperties(size=20),
    #     loc=2)
    #
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.title(clf_name)
    # plt.show()
