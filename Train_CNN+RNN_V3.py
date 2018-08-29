# CNN + RNN on 20 images
import tensorflow as tf
import random
import numpy as np
import random

# hyper parameters
learning_rate = 0.0001
training_epochs = 400
batch_size = 100
final_output = 7

# parameters for rnn
time_step_size = 20 # which is the number of frames
lstm_size = 3072

tf.set_random_seed(777)  # reproducibility

data_all = np.loadtxt('image_recognition_data.csv', delimiter=',', dtype=np.float32)
data = data_all[:5000]
data_test = data_all[5000:]


def next_batch(step):
    if ((step + 1) * batch_size > len(data)):
        data_piece = data[step * batch_size:]
    else:
        data_piece = data[step * batch_size:(step + 1) * batch_size]
    Ys_raw = data_piece[:, [-1]]
    random.shuffle(data_piece)
    Ys = []
    for i in range(len(Ys_raw)):
        entry = [0] * 7
        for result in range(7):
            if (Ys_raw[i][0] == result):
                entry[result] = 1
            else:
                entry[result] = 0
        Ys.append(entry)

    return data_piece[:, :-1], Ys

def test_set() :
    size = np.shape(data_test)[0]
    temp_test = np.asarray([data_test[random.randint(0,size-1)]])
    for i in range(99) :
        temp_test = np.concatenate((temp_test, [data_test[random.randint(0,size-1)]]), axis = 0)
    Ys_raw = temp_test[:, [-1]]
    Ys = []
    for i in range(len(Ys_raw)):
        entry = [0] * 7
        for result in range(7):
            if (Ys_raw[i][0] == result):
                entry[result] = 1
            else:
                entry[result] = 0
        Ys.append(entry)
    return temp_test[:, :-1], Ys
    


# < Input Process >
# X : [batch_size, 20 * 48 * 48 * 1]
# (processed) --> X_images : [20, batch_size, 48, 48, 1]
# Y : [batch_size, 7]`
X = tf.placeholder(tf.float32, [None, 20 * 48 * 48 * 1])
X_reshaped = tf.reshape(X, [-1, 20, 48, 48, 1])  # img 48*48*1 (gray)
X_images = []
for i in range(20):
    temp = tf.slice(X_reshaped, [0, i, 0, 0, 0], [-1, 1, -1, -1, -1])
    X_images.append(tf.reshape(temp,  [-1, 48, 48, 1]))
Y = tf.placeholder(tf.float32, [None, 7])



# < Convolution + MaxPool 1 : Layer1 >
# (+) Conv1    : [?, 48, 48, 1] --> [?, 44, 44, 64]
# (+) MaxPool1 : [?, 44, 44, 64] --> [?, 22, 22, 64]
# X_images_after_L1 : [20, ?, 22, 22, 64]
W1 = tf.get_variable("W1", shape=[5, 5, 1, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
X_images_after_L1 = [0]*20
for i in range(20):
    temp = tf.nn.conv2d(X_images[i], W1, strides=[1, 1, 1, 1], padding='VALID')
    temp = tf.nn.relu(temp)
    temp = tf.nn.max_pool(temp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    X_images_after_L1[i] = temp



# < Convolution + MaxPool 2 : Layer2 >
# (+) Conv2    : [?, 22, 22, 64] --> [?, 18, 18, 64]
# (+) MaxPool2 : [?, 18, 18, 64] --> [?, 9, 9, 64]
# X_images_after_L2 : [20, ?, 9, 9, 64]
W2 = tf.get_variable("W2", shape=[5, 5, 64, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
X_images_after_L2 = [0]*20
for i in range(20):
    temp = tf.nn.conv2d(X_images_after_L1[i], W2, strides=[1, 1, 1, 1], padding='VALID')
    temp = tf.nn.relu(temp)
    temp = tf.nn.max_pool(temp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    X_images_after_L2[i] = temp



# < Convolution (No MaxPool) : Layer3 >
# (+) Conv3 : [?, 9, 9, 64] --> [?, 6, 6, 128]
# X_images_after_L3 : [20, ?, 6 * 6 * 64] (flattened for layer4 (which is fully connected layer)
W3 = tf.get_variable("W3", shape=[4, 4, 64, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
X_images_after_L3 = [0]*20
for i in range(20):
    temp = tf.nn.conv2d(X_images_after_L2[i], W3, strides=[1, 1, 1, 1], padding='VALID')
    temp = tf.nn.relu(temp)
    temp = tf.reshape(temp, [-1, 6 * 6 * 128])
    X_images_after_L3[i] = temp



# < Fully Connected to 3072 nodes : Layer4 >
# X_images_after_L4 : [?, 20, 3072]
W4 = tf.get_variable("W4", shape=[6 * 6 * 128, 3072],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", shape=[3072],
                     initializer=tf.contrib.layers.xavier_initializer())

X_images_after_L4 = tf.reshape(tf.matmul(X_images_after_L3[0], W4) + b4, [-1, 1, 3072])
for i in range(1, 20, 1):
    temp = tf.reshape(tf.matmul(X_images_after_L3[i], W4) + b4, [-1, 1, 3072])
    X_images_after_L4 = tf.concat([X_images_after_L4, temp], 1)
    


# < RNN on features-extracted images : Layer5 >
# logits : [?, 3072]

W5 = tf.get_variable("W5", shape=[20*3072, 7],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable("b5", shape=[7],
                     initializer=tf.contrib.layers.xavier_initializer())
    # X, input shape: (?, 20, 3072)
XT = tf.transpose(X_images_after_L4, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (20, ?, 3072)
XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
X_split = tf.split(XR, time_step_size, 0)
    # Each array shape: (?, 3072)
    # Make lstm with lstm_size


lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    # Get lstm cell output, 20 arrays with lstm_size output: (?, 3072)
outputs, _states = tf.nn.static_rnn(lstm, X_split, dtype=tf.float32)
    # Linear activation
    # Get the last output

outputs_reshaped = tf.reshape(outputs[0], [-1, 1, 3072])
for i in range(1,20,1) :
    outputs_reshaped = tf.concat([outputs_reshaped, tf.reshape(outputs[i], [-1, 1, 3072])], 1)

logits, _state = tf.matmul(tf.reshape(outputs_reshaped, [-1, 20*3072]), W5) + b5, lstm.state_size # State size to initialize the stat

# hypothesis used for later measurement of accuracy
hypothesis = tf.nn.softmax(logits)


# define cost/loss & optimizer
print("logits : ", logits)
print("Y : ", Y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()


# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# train my model
print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = (int)(np.ceil(float(len(data)) / batch_size))

    for step in range(total_batch):
        feed_X, feed_Y = next_batch(step)
        feed_dict = {X: feed_X, Y: feed_Y}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    feed_X_test, feed_Y_test = test_set()
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))

    a = sess.run(acc, feed_dict = {X: feed_X_test, Y: feed_Y_test})

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Accuracy: ', a)

save_path = saver.save(sess, "./version_3.ckpt")

print('Learning Finished!')
