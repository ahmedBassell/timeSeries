"""Simple tutorial for using TensorFlow to compute polynomial regression.
Parag K. Mital, Jan. 2016"""
# %% Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import normalize

# read db
# 
db_path = os.path.join(os.path.dirname(__file__), 'db/newdb.csv')
df = pd.read_csv(db_path, sep=',', dtype={'id':np.int64, 'hour':np.int64, 'camid': np.int64})


# x = df[['Date', 'day=Friday', 'day=Saturday', 'day=Sunday', 'day=Monday', 'day=Tuesday', 'day=Wednesday', 'day=Thursday']]
ys = df['count'].as_matrix().reshape(-1,1)


# ys = ys / np.linalg.norm(ys)
# norm2 = normalize(x[:,np.newaxis], axis=0).ravel()


n_observations = len(ys)
# print ys


# %% Let's create some toy data
plt.ion()
# n_observations = 100
fig, ax = plt.subplots(1, 1)


xs = np.linspace(0, 1, n_observations)
xs_pr = np.linspace(0, 2, 2*n_observations)
# ys = np.sin(np.pi*xs) + np.random.uniform(-0.5, 0.5, n_observations) + xs







ax.scatter(xs, ys)
fig.show()
plt.draw()

# %% tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph.
X = tf.placeholder(tf.float64)
Y = tf.placeholder(tf.float64)

# %% Instead of a single factor and a bias, we'll create a polynomial function
# of different polynomial degrees.  We will then learn the influence that each
# degree of the input (X^0, X^1, X^2, ...) has on the final output (Y).
Y_pred = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='bias', dtype=tf.float64)
for pow_i in range(1, 5):
    W1 = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='weight1_%d' % pow_i, dtype=tf.float64)
    Y_pred = tf.add(tf.mul(tf.pow(X, pow_i), W1), Y_pred)

for interval in range(1,24):
    W2 = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='weight2_%d' % pow_i, dtype=tf.float64)
    Y_pred = tf.add(tf.mul(tf.sin(2*np.pi*X*n_observations*2 /(interval)), W2), Y_pred)

for interval in range(1,24):
    W4 = tf.Variable(tf.random_normal([1], dtype=tf.float64), name='weight3_%d' % pow_i, dtype=tf.float64)
    Y_pred = tf.add(tf.mul(tf.mul(tf.sin(2*np.pi*X*n_observations*2 /(interval)), X), W4), Y_pred)

# %% Loss function will measure the distance between our observations
# and predictions and average over them.
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

# %% if we wanted to add regularization, we could add other terms to the cost,
# e.g. ridge regression has a parameter controlling the amount of shrinkage
# over the norm of activations. the larger the shrinkage, the more robust
# to collinearity.
# cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam').minimize(cost)

# %% We create a session to use the graph
n_epochs = 500
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.initialize_all_variables())

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(
            cost, feed_dict={X: xs, Y: ys})
        print(epoch_i, training_cost)

        # if epoch_i % 100 == 0:
        #     # ax.clear()
        #     ax.plot(xs, Y_pred.eval(feed_dict={X: xs}, session=sess),'k', alpha=epoch_i / n_epochs)
        #     # fig.show()
        #     plt.draw()
        #     plt.pause(0.01)

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

    ax.plot(xs_pr, Y_pred.eval(feed_dict={X: xs_pr}, session=sess),'k')
    fig.show()
    plt.draw()
ax.set_ylim([0, 40000])
fig.show()
plt.waitforbuttonpress()