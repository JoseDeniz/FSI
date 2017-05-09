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


def read_data_from_filename(filename):
    data = np.genfromtxt(filename, delimiter=",")  # data file loading
    np.random.shuffle(data)  # we shuffle the data

    x_data_train = data[0:107, 0:4].astype('f4')  # the samples are the four first rows of data
    y_data_train = one_hot(data[0:107, 4].astype(int),
                           3)  # the labels are in the last row. Then we encode them in one hot code

    x_data_validation = data[107:129, 0:4].astype('f4')  # the samples are the four first rows of data
    y_data_validation = one_hot(data[107:129, 4].astype(int),
                                3)  # the labels are in the last row. Then we encode them in one hot code

    x_data_test = data[129:151, 0:4].astype('f4')
    y_data_test = one_hot(data[129:151, 4].astype(int), 3)
    return x_data_train, y_data_train, x_data_validation, y_data_validation, x_data_test, y_data_test


def print_results(mode, error, batch_xs, batch_ys, epoch_number=1):
    print mode, " epoch #:", epoch_number, "Error: ", error
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r


# Training & validation
x_training_data, y_training_data, x_validation_data, y_validation_data, x_test_data, y_test_data = read_data_from_filename(
    filename='iris.data')

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
training_errors = []
validation_errors = []
last_validation_error = 1
validation_error = 0.1
epoch = 0
difference = 100.0

while validation_error <= last_validation_error and difference > 0.001:
    epoch += 1

    for jj in xrange(len(x_training_data) / batch_size):
        batch_training_xs = x_training_data[jj * batch_size: jj * batch_size + batch_size]
        batch_training_ys = y_training_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_training_xs, y_: batch_training_ys})

    # Training
    training_error = sess.run(loss, feed_dict={x: batch_training_xs, y_: batch_training_ys})
    training_errors.append(training_error)

    # Validation
    validation_error = sess.run(loss, feed_dict={x: x_validation_data, y_: y_validation_data})

    validation_errors.append(validation_error)
    if epoch > 1:
        difference = validation_errors[-2] - validation_error
    last_validation_error = validation_errors[-1]
    # Training
    print_results(mode="Training", error=training_error, batch_xs=batch_training_xs, batch_ys=batch_training_ys,
                  epoch_number=epoch)

    # Validation
    print_results(mode="Validation", error=validation_error, batch_xs=x_validation_data, batch_ys=y_validation_data,
                  epoch_number=epoch)
    print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Training finished  "
print "----------------------"


print "----------------------"
print "   Start testing...  "
print "----------------------"

sess.run(train, feed_dict={x: x_test_data, y_: y_test_data})
test_error = sess.run(loss, feed_dict={x: x_test_data, y_: y_test_data})

print_results(mode="Testing", error=test_error, batch_xs=x_test_data, batch_ys=y_test_data)

print "----------------------------------------------------------------------------------------"
print "   Testing finished: error ", test_error, ", last validation error ", validation_errors[-1]
print "----------------------------------------------------------------------------------------"

# Plot
plt.ylabel('Errors')
plt.xlabel('Epochs')
training_line, = plt.plot(training_errors)
validation_line, = plt.plot(validation_errors)
plt.legend(handles=[training_line, validation_line],
           labels=["Training errors", "Validation errors"])
plt.savefig('training_vs_validation_plot.png')