import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

# Training
training_data = np.genfromtxt('training.data', delimiter=",")  # training.data file loading
np.random.shuffle(training_data)  # we shuffle the data
x_training_data = training_data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_training_data = one_hot(training_data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# Validation
validation_data = np.genfromtxt('validation.data', delimiter=",")  # validation.data file loading
np.random.shuffle(validation_data)  # we shuffle the data
x_validation_data = validation_data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_validation_data = one_hot(validation_data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# print "\nSome samples..."
# for i in range(20):
#     print x_training_data[i], " -> ", y_training_data[i]
# print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
maximum_validation_errors = 8
validation_error_counter = 0
training_errors = []
validation_errors = []

for epoch in xrange(1000):

    for jj in xrange(len(x_training_data) / batch_size):
        batch_training_xs = x_training_data[jj * batch_size: jj * batch_size + batch_size]
        batch_training_ys = y_training_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_training_xs, y_: batch_training_ys})

    for kk in xrange(len(x_validation_data) / batch_size):
        batch_validation_xs = x_validation_data[kk * batch_size: kk * batch_size + batch_size]
        batch_validation_ys = y_validation_data[kk * batch_size: kk * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_validation_xs, y_: batch_validation_ys})

    # Training
    training_error = sess.run(loss, feed_dict={x: batch_training_xs, y_: batch_training_ys})
    training_errors.append(training_error)

    # Validation
    validation_error = sess.run(loss, feed_dict={x: batch_validation_xs, y_: batch_validation_ys})

    if validation_errors:
        last_validation_error = validation_errors[-1]
        if validation_error >= last_validation_error:
            validation_error_counter += 1
            if validation_error_counter > maximum_validation_errors:
                print "Exceeded maximum number[%d] of validation errors upticks, " \
                      "so the training is stopped in epoch %d\n errors: %s" \
                      % (maximum_validation_errors, epoch, validation_errors)
                break
        else:
            validation_error_counter = 0
    validation_errors.append(validation_error)

    # Training
    # print "Training epoch #:", epoch, "Error: ", training_error
    # result = sess.run(y, feed_dict={x: batch_training_xs})
    # for b, r in zip(batch_training_ys, result):
    #     print b, "-->", r

    # Validation
    # print "Validation epoch #:", epoch, "Error: ", validation_error
    # result = sess.run(y, feed_dict={x: batch_validation_xs})
    # for b, r in zip(batch_validation_ys, result):
    #     print b, "-->", r
    # print "----------------------------------------------------------------------------------"

plt.ylabel('Errors')
plt.xlabel('Epochs')
training_line, = plt.plot(training_errors)
validation_line, = plt.plot(validation_errors)
plt.legend(handles=[training_line, validation_line],
           labels=["Training errors", "Validation errors"])
plt.savefig('training_vs_validation_plot.png')
