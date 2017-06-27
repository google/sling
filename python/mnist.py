from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from flow import Flow
from flow import FlowBuilder

# Import data.
mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)

# Create the model.
x = tf.placeholder(tf.float32, [None, 784], name='x')
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
y = tf.add(tf.matmul(x, W), b, name='y')

# Define loss and optimizer.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Train model.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Save model to flow file.
flow = Flow()
builder = FlowBuilder(sess, flow)
builder.add(flow.func("classifier"), [x], [y])
flow.save("/tmp/mnist.flow")

