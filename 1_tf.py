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
warnings.filterwarnings('ignore')

tic = time.time()

data = pd.read_excel("Signal Detection.xlsx")

df_siteid = data['Site ID']

# df_new_onehot = df_new.copy()

c = ['SF Ratio', 'PD Ratio', 'AE Ratio', 'SAE Ratio', 'discontinued patients Ratio']
# d = [['0x4', '0x41', '0x10', '0x0'], ['Command', 'X'], ['OffControlMode', 'X', 'AutoControlMode'],
#      ['SolenoidControlScheme', 'X'], ['0x10', '0x3', '0x0'], ['X', 'InvalidDataLength'], ['X', 'InvalidFunctionCode']]

# # indicator cols
# for i in range(len(c)):
#     a1 = pd.get_dummies(df_new_onehot, c[i])
#
# data= data.copy()
# data= data.drop(['Address', 'PIDRate', 'PipelinePSI', 'deltaPIDRate', 'deltaPipelinePSI', 'CommandResponse',
#                  'ControlMode', 'ControlScheme', 'FunctionCode', 'InvalidDataLength', 'InvalidFunctionCode',
#                  'PumpState', 'SolenoidState'], axis=1)

# df1 = data.copy()
# df1_y = df1['Label']
df1 = pd.DataFrame(data['SF Ratio'])
df1 = df1.join(data['PD Ratio'])
df1 = df1.join(data['AE Ratio'])
df1 = df1.join(data['SAE Ratio'])
df1 = df1.join(data['discontinued patients Ratio'])

df1 = df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


df1 = normalize(df1)
# df1 = df1.join(df1_y)
df1 = df1.join(df_siteid)
data = df1.copy()

# print('data: ', data.head())
# print('length: ', len(data))

# data = data.join(a1)

# count_classes = pd.value_counts(df['Label'], sort=True).sort_index()


class Autoencoder(object):

    def __init__(self, n_hidden_1, n_hidden_2, n_input, learning_rate):
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_input = n_input

        self.learning_rate = learning_rate

        self.weights, self.biases = self._initialize_weights()

        self.x = tf.placeholder("float", [None, self.n_input])

        self.encoder_op = self.encoder(self.x)
        self.decoder_op = self.decoder(self.encoder_op)

        self.cost = tf.reduce_mean(tf.pow(self.x - self.decoder_op, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
        }

        return weights, biases

    def encoder(self, X):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(X, self.weights['encoder_h1']),
                                    self.biases['encoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                    self.biases['encoder_b2']))
        return layer_2

    def decoder(self, X):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(X, self.weights['decoder_h1']),
                                    self.biases['decoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                    self.biases['decoder_b2']))
        return layer_2

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        return self.sess.run(self.encoder_op, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.decoder_op, feed_dict={self.x: X})


# good_data = data[data['Label'] == 0]
#
# bad_data = data[data['Label'] == 1]
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

X_train_siteid = X_train['Site ID']
X_train = X_train.drop(['Site ID'], axis=1)

X_test_siteid = pd.DataFrame(X_test['Site ID'])
X_test = X_test.drop(['Site ID'], axis=1)

# y_test = X_test['Label']
# X_test = X_test.drop(['Label'], axis=1)
#
# X_train = X_train.values
# X_test = X_test.values

# X_good = good_data.ix[:, good_data.columns != 'Label']
# y_good = good_data.ix[:, good_data.columns == 'Label']
# X_bad = bad_data.ix[:, bad_data.columns != 'Label']
# y_bad = bad_data.ix[:, bad_data.columns == 'Label']

model = Autoencoder(n_hidden_1=2, n_hidden_2=1, n_input=X_train.shape[1], learning_rate= 0.0001)

training_epochs = 100
batch_size = 128
display_step = 10
record_step = 10

total_batch = int(X_train.shape[0] / batch_size)

cost_summary = []

for epoch in range(training_epochs):
    cost = None
    for i in range(total_batch):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = X_train.iloc[batch_start:batch_end, :]

        cost = model.partial_fit(batch)

    if epoch % display_step == 0 or epoch % record_step == 0:
        total_cost = model.calc_total_cost(X_train)

        if epoch % record_step == 0:
            cost_summary.append({'epoch': epoch + 1, 'cost': total_cost})

        if epoch % display_step == 0:
            print("Epoch:{}, cost={:.9f}".format(epoch + 1, total_cost))

encode_decode = None
total_batch = int(X_test.shape[0] / batch_size) + 1

for i in range(total_batch):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch = X_test.iloc[batch_start:batch_end, :]
    batch_res = model.reconstruct(batch)
    if encode_decode is None:
        encode_decode = batch_res
    else:
        encode_decode = np.vstack((encode_decode, batch_res))


def get_df(orig, ed):
    rmse = np.mean(np.power(orig - ed, 2), axis=1)
    return pd.DataFrame({'rmse': rmse})


df = get_df(X_test, encode_decode)
#print(df.rmse.values)
# print('rmse: ', df)
# print('length: ', len(df))

# opt_threshold = 0
# df= normalize(df)
# print("============================")
# print(df.info())
# print("+++++++++++++++++++++++++++")
print(df.rmse.values)


#print(centroids)
#print(centroids.size)
#print(df.size)
y=np.array(129,dtype=float)
y= pd.DataFrame(y)
y.columns=['y']

df=pd.DataFrame(df)
df=df.join(y)
df['y']=1

kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_

plt.scatter(df['rmse'],df['y'] , c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:,0], centroids[:, 1], c=['red','blue'], s=50)
plt.show()
y_pred = [1 if p > 0.5 else 0 for p in df.rmse.values]

#print(df.rmse.values)

# y_pred = normalize(pd.DataFrame(y_pred))


# cnf_matrix = confusion_matrix(df.target, y_pred)s
# cnf_matrix1 = confusion_matrix(df.target, y_pred)
# opt_acc = float((cnf_matrix[0, 0] + cnf_matrix[1, 1]) /
#                 (cnf_matrix[1, 0] + cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[0, 0]))

# for i in range(int(100 * np.max(df.rmse))):
#     y_pred = [1 if p > float(i/100) else 0 for p in df.rmse.values]
#     cnf_matrix = confusion_matrix(df.target, y_pred)
#     temp_acc = float((cnf_matrix[0, 0] + cnf_matrix[1, 1]) /
#                      (cnf_matrix[1, 0] + cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[0, 0]))
#     np.set_printoptions(precision=2)
#     if temp_acc > opt_acc:
#         opt_acc = temp_acc
#         opt_threshold = float(i/100)
#         cnf_matrix1 = cnf_matrix

zeros = 0
ones = 0
for i in y_pred:
    if i == 0:
        zeros = zeros+1
    elif i == 1:
        ones = ones+1


# print('zeros: ', zeros)
# print('ones: ', ones)
#
# print('df_siteid: ', len(X_test_siteid))
# print('y_pred: ', len(y_pred))

set_siteid = []
X_test_siteid = X_test_siteid['Site ID'].tolist()
for i in range(len(y_pred)):
    if y_pred[i] == 0:
        set_siteid.append(X_test_siteid[i])

print('Sites with anomaly: ', set_siteid)

# print('optimal threshold = ', opt_threshold)

# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     else:
#         1
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

#
# class_names = [0, 1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix1, classes=class_names, title='Confusion matrix')
# plt.show()
