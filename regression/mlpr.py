

# import tensorflow as tf

# # Import MINST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/home/ahmed/Desktop/companies_projects/avidbeams/tensorflowCondaEnv/xxx/", one_hot=True)


# # Parameters
# learning_rate = 0.01
# training_epochs = 25
# batch_size = 100
# display_step = 1

# # tf Graph Input
# x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
# y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# # Set model weights
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))

# # Construct model
# pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# # Minimize error using cross entropy
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# # Gradient Descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# # Initializing the variables
# init = tf.initialize_all_variables()






# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# print trX
# print trY

# # # Launch the graph
# # with tf.Session() as sess:
# #     sess.run(init)

# #     # Training cycle
# #     for epoch in range(training_epochs):
# #         avg_cost = 0.
# #         total_batch = int(mnist.train.num_examples/batch_size)
# #         # Loop over all batches
# #         for i in range(total_batch):
# #             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# #             # Fit training using batch data
# #             _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
# #                                                           y: batch_ys})
# #             # Compute average loss
# #             avg_cost += c / total_batch
# #         # Display logs per epoch step
# #         if (epoch+1) % display_step == 0:
# #             print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

# #     print "Optimization Finished!"

# #     # Test model
# #     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# #     # Calculate accuracy for 3000 examples
# #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# #     print "Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]})
# 











import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.05
training_epochs = 50

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy












import numpy as np
import pandas as pd
import os

# read db
db_path = os.path.join(os.path.dirname(__file__), 'db/newdb.csv')
df = pd.read_csv(db_path, sep=',', dtype={'id':np.int64, 'hour':np.int64, 'camid': np.int64})

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


# feature and target
x = df[['Date', 'day=Friday', 'day=Saturday', 'day=Sunday', 'day=Monday', 'day=Tuesday', 'day=Wednesday', 'day=Thursday']]
y = df['count'].as_matrix().reshape(-1,1)


# Split the data into training/testing sets
x_train = x[:-20]
x_test = x[-20:]

# Split the targets into training/testing sets
y_train = y[:-20]
y_test = y[-20:]


x_train = x_train.as_matrix()
x_test = x_test.as_matrix()

# y_train = y_train.as_matrix()
# y_test = y_test.as_matrix()


# print x_train.as_matrix()
# print type(x_train.as_matrix())









# # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 8]) # create symbolic variables
Y = tf.placeholder("float", [None, 1])

w = init_weights([8, 1]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(training_epochs):
        for start, end in zip(range(0, len(x_train), 128), range(128, len(x_train), 128)):
            sess.run(train_op, feed_dict={X: x_train[start:end], Y: y_train[start:end]})
        # print(i, np.mean(np.argmax(y_test, axis=1) ==
        #                  sess.run(predict_op, feed_dict={X: x_test, Y: y_test})))

        # print ( sess.run(predict_op, feed_dict={X: x_test, Y: y_test})
        