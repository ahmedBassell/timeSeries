
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
print "range: ",rng.rand()
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
# train_X = numpy.asarray([ 736087,  736088,  736089,  736090,  736091,  736092,  736093,  736094,
#   736095,  736096,  736097,  736098,  736099,  736100,  736101,  736102,
#   736103,  736104,  736105,  736106,  736107,  736108,  736109,  736110,
#   736111,  736112,  736113,  736114,  736115,  736116,  736117,  736118,
#   736119,  736120,  736121])
# train_Y = numpy.asarray([3605,   4745,   1446,    149,    355,   1390,   5907,  10713,  14815,
#   15761,   2314,   3783,  17278,  16867,  17088,  14821,  17654,   2079,
#    3551,  17715,  17686,  15548,  15535,  16012,   2271,   3425,  13371,
#   16789,  16106,  16971,  15954,   2489,   2288,   3098,   2665,])

# print type(train_X[0])
# print train_X


import numpy as np
import pandas as pd
import os

# read db
db_path = os.path.join(os.path.dirname(__file__), 'db/newdb.csv')
df = pd.read_csv(db_path, sep=',', dtype={'id':np.int64, 'hour':np.int64, 'camid': np.int64, 'Date':np.float64, 'count': np.float64})



# enc = preprocessing.OneHotEncoder()

# enc.fit([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 0, 1],])  

# OneHotEncoder(categorical_features='day', dtype=<'str'>,handle_unknown='error', n_values='auto', sparse=True)

# enc.transform([[0, 1, 3]]).toarray()




from sklearn.feature_extraction import DictVectorizer

def one_hot_dataframe(data, cols, replace=False):
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

# data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
#         'year': [2000, 2001, 2002, 2001, 2002],
#         'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

# df = pd.DataFrame(data)

df2, _, _ = one_hot_dataframe(df, ['day'], replace=True)
# print df2
df = df2



# x = df[['Date', 'day=Friday', 'day=Saturday', 'day=Sunday', 'day=Monday', 'day=Tuesday', 'day=Wednesday', 'day=Thursday']]

x = df['Date'].as_matrix()
y = df['count'].as_matrix()


# Split the data into training/testing sets
train_X = x[:-20]
test_X = x[-20:]

# Split the targets into training/testing sets
train_Y = y[:-20]
test_Y = y[-20:]

# print type(train_X[0])
print np.log(train_X)

train_X = np.log(train_X)
train_Y = np.log(train_Y)

test_X = np.log(test_X)
test_Y = np.log(test_Y)







n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print sess.run(W)
            print sess.run(b)
            print sess.run(pred, feed_dict={X: train_X})
            # print train_X
            # print train_Y
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    # test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    # test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print "Testing... (Mean square loss Comparison)"
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print "Testing cost=", testing_cost
    print "Absolute mean square loss difference:", abs(
        training_cost - testing_cost)

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()