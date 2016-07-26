import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

rng = np.random
# print rng.randn()
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# read db
db_path = os.path.join(os.path.dirname(__file__), 'db/newdb.csv')
df = pd.read_csv(db_path, sep=',', dtype={'id':np.int64, 'hour':np.int64, 'camid': np.int64})

# feature and target
df_x = df['Date'].as_matrix()
df_y = df['count'].as_matrix()

df_y = df_y.astype(np.float64)
df_x = df_x.astype(np.float64)
# Split the data into training/testing sets
train_X = df_x[:-20]
test_X = df_x[-20:]
# Split the targets into training/testing sets
train_Y = df_y[:-20]
test_Y = df_y[-20:]


# # Training Data
# train_X = np.asarray([1,2,3,4,5,6,7,8,9,10])
# train_Y = np.asarray([1,2,3,4,5,6,7,8,9,10])
n_samples = train_X.shape[0]

print type(train_Y)
print train_Y

print type(train_X)
print train_X

print type(train_X[0])

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
    # test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    # test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

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