#tf.__version__ >=1.3
import tensorflow as tf
#一、Matrix operations
#1.simple mathematical operation takes an addition as example
mat = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
add_row = tf.reduce_sum(mat, 1)
add_col = tf.reduce_sum(mat, 0)
session = tf.Session()
assert add_row.eval(session=session).all() == np.array([6, 15, 24, 33]).all()
assert add_col.eval(session=session).all() == np.array([22, 26, 30]).all()

#二、Variables and operations
#1. variable and get_variable
#use tf.Variable the var can have the same name but not with get_variable
var1 = tf.Variable([1], name="var")
with tf.variable_scope("foo"):
  with tf.variable_scope("bar"):
    var2 = tf.Variable([1], name="var")
    var3 = tf.Variable([1], name="var")

print("var1: {}".format(var1.name))
print("var2: {}".format(var2.name))
print("var3: {}".format(var3.name))

#var1: var:0
#var2: foo/bar/var:0
#var3: foo/bar/var_1:0

#with respect to fet_vaeiable set reuse
#when the variable name conflict,with a similar var_1, var_2 ...instead
var1 = tf.get_variable(name="var",shape=[1])
with tf.variable_scope("foo"):
  with tf.variable_scope("bar") as scp:
    var2 = tf.get_variable(name="var",shape=[1])
    scp.reuse_variables()
    var3 = tf.get_variable(name="var",shape=[1])

print("var1: {}".format(var1.name))
print("var2: {}".format(var2.name))
print("var3: {}".format(var3.name))

#var1: var_1:0
#var2: foo/bar/var_2:0
#var3: foo/bar/var_2:0

#the same variable in different graph can have the same name
g1 = tf.Graph()
with g1.as_default():
  with tf.variable_scope("foo"):
    with tf.variable_scope("bar") as scp:
      var1 = tf.get_variable(name="var",shape=[1])

g2 = tf.Graph()
with g2.as_default():
  with tf.variable_scope("foo"):
    with tf.variable_scope("bar") as scp:
      var2 = tf.get_variable(name="var",shape=[1])

print("var1: {}".format(var1.name))
print("var2: {}".format(var2.name))

#var1: foo/bar/var:0
#var2: foo/bar/var:0

#三、Recurrent Neural Networks(RNN)
#1.build LSTM(GRU) cell
cell = tf.contrib.rnn.BasicLSTMCell(num_units=32, forget_bias=0.0)

#2.build rnn
#dynamic_rnn If there is no initial_state, you must give a dtype.
#as seen in session, feed_dict is not necessary
import numpy as np
rnn_input = np.random.randn(2, 6, 8)
rnn_input[1,4:] = 0
sequence_length = [6, 4]
outputs, state = tf.nn.dynamic_rnn(
  cell=cell,
  inputs=rnn_input,
  dtype=tf.float64,
  sequence_length=sequence_length)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs, state = sess.run([outputs, state], feed_dict=None)

assert outputs.shape == (2, 6, 32)
assert outputs[1,4,:].all() == np.zeros(cell.output_size).all()

#3.build bidirectional rnn
outputs, state = tf.nn.bidirectional_dynamic_rnn(
  cell_fw=cell,
  cell_bw=cell,
  dtype=tf.float64,
  sequence_length=sequence_length,
  inputs=rnn_input)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs, state = sess.run([outputs, state], feed_dict=None)

output_fw, output_bw = outputs
state_fw, state_bw = state

print(output_fw.shape)
print(output_bw.shape)
print(state_fw.h.shape)
print(state_bw.h.shape)

#(2, 6, 32)
#(2, 6, 32)
#(2, 32)
#(2, 32)

#4.get dynamic_rnn last time output
import numpy as np
rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 1]

cell = tf.contrib.rnn.LSTMCell(num_units=32)
cell_state = cell.zero_state(len(sequence_length), tf.float32)
outputs, state = tf.nn.dynamic_rnn(
  cell=cell,
  inputs=rnn_input,
  sequence_length=sequence_length,
  dtype=tf.float32,
  initial_state=cell_state)

#because tensorflow doesn't support advanced slice, so we cant't get the last relevent
# by outputs[:, length - 1] like numpy, if we run first, then we can't train end-to-end
def get_last_relevent(outputs, length):
  if not isinstance(length, np.ndarray):
    length = np.array(length)

  batch_size, max_length, hidden_size = outputs.get_shape().as_list()
  index = [max_length * i for i in range(batch_size)] + (length - 1)
  outputs_flat = tf.reshape(outputs, [-1, hidden_size])
  last_relevent = tf.gather(outputs_flat, index)
  return last_relevent

relevant = get_last_relevent(outputs, sequence_length)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs, state, last_relevent = sess.run([outputs, state, relevant], feed_dict=None)

assert state.h.all() == last_relevent.all()