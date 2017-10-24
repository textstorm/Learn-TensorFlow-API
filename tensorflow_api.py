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
#can use cell can also use cell_fw, cell_bw(nmt used)
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

cell = tf.contrib.rnn.LSTMCell(num_units=8)
cell_state = cell.zero_state(len(sequence_length), tf.float32)
outputs, state = tf.nn.dynamic_rnn(
  cell=cell,
  inputs=rnn_input,
  sequence_length=sequence_length,
  dtype=tf.float32,
  initial_state=cell_state)

#because tensorflow doesn't support advanced slice, so we cant't get the last relevant
# by outputs[:, length - 1] like numpy, if we run first, then we can't train end-to-end
def get_last_relevant(outputs, length):
  if not isinstance(length, np.ndarray):
    length = np.array(length)

  batch_size, max_length, hidden_size = outputs.get_shape().as_list()
  index = [max_length * i for i in range(batch_size)] + (length - 1)
  outputs_flat = tf.reshape(outputs, [-1, hidden_size])
  last_relevant = tf.gather(outputs_flat, index)
  return last_relevant

relevant = get_last_relevant(outputs, sequence_length)
#relevant_ = outputs[:,:sequence_length,:] #can't fetch last relevant in this way
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs, state, last_relevant = sess.run([outputs, state, relevant], feed_dict=None)

assert state.h.all() == last_relevant.all()

#5.get bidirectional dynamic_rnn last time output
rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 1]

cell = tf.contrib.rnn.LSTMCell(num_units=8)
cell_state = cell.zero_state(len(sequence_length), tf.float32)
outputs, state = tf.nn.bidirectional_dynamic_rnn(
  cell_fw=cell,
  cell_bw=cell,
  inputs=rnn_input,
  sequence_length=sequence_length,
  dtype=tf.float32,
  initial_state_fw=cell_state,
  initial_state_bw=cell_state)

outputs_concated = tf.concat(outputs, 2)
state_concated = tf.concat(state, 2)

def get_last_relevant(outputs_concated, length):
  if not isinstance(length, np.ndarray):
    length = np.array(length)

  batch_size, max_length, hidden_size = outputs_concated.get_shape().as_list()
  index = [max_length * i for i in range(batch_size)] + (length - 1)
  outputs_flat = tf.reshape(outputs_concated, [-1, hidden_size])
  last_relevant = tf.gather(outputs_flat, index)
  return last_relevant

relevant = get_last_relevant(outputs_concated, sequence_length)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_, outputs_concated, state_concated, last_relevant = sess.run(
    [outputs, outputs_concated, state_concated, relevant], feed_dict=None)

assert state_concated[-1,:,:].all() == last_relevant.all()

#5.multi-layers rnn test
import tensorflow as tf
import numpy as np
rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 1]

cell_list = []
num_layers = 3
for _ in range(num_layers):
  cell = tf.contrib.rnn.LSTMCell(num_units=8)
  cell_list.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cell_list)
cell_state = cell.zero_state(len(sequence_length), tf.float32)

outputs, state = tf.nn.dynamic_rnn(
  cell=cell,
  inputs=rnn_input,
  sequence_length=sequence_length,
  initial_state=cell_state)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_, state_ = sess.run([outputs, state], feed_dict=None)

#multi-layers process can be summarized as
def build_rnn_cell(num_units, num_layers):
  cell_list = []
  for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(num_units)
    cell_list.append(cell)
  if num_layers == 1:
    return cell_list[0]
  else:
    return tf.contrib.rnn.MultiRNNCell(cell_list)

#6.multi-layers bidirectional rnn
import tensorflow as tf
import numpy as np
rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 1]
num_layers = 2

def build_rnn_cell(num_units, num_layers):
  cell_list = []
  for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(num_units)
    cell_list.append(cell)
  if num_layers == 1:
    return cell_list[0]
  else:
    return tf.contrib.rnn.MultiRNNCell(cell_list)

cell_fw = build_rnn_cell(8, num_layers)
cell_bw = build_rnn_cell(8, num_layers)
outputs, state = tf.nn.bidirectional_dynamic_rnn(
  cell_fw=cell_fw, 
  cell_bw=cell_bw,
  inputs=rnn_input, 
  dtype=tf.float32,
  sequence_length=sequence_length)

outputs_fw, output_bw = outputs
state_fw, state_bw = state
print(outputs_fw.shape)
print(output_bw.shape)
#(2, 4, 8)
#(2, 4, 8)

outputs_concated = tf.concat(outputs, 2)
print(state)
#the last state of each layer (fw_or_bw, layers)
#((LSTMStateTuple(c=<>, h=<>),  fw_1
#  LSTMStateTuple(c=<>, h=<>)), fw_2
# (LSTMStateTuple(c=<>, h=<>),  bw_1
#  LSTMStateTuple(c=<>, h=<>))) bw_2

state_concated = []
for layer_id in range(num_layers):
  state_concated.append(state[0][layer_id])
  state_concated.append(state[1][layer_id])

state_concated = tuple(state_concated)

#7.state test (important)
#in bidirection rnn the 1 layer include 2 layers rnn
#test one layer bidirection rnn(two layers)
import tensorflow as tf
import numpy as np
num_layers = 2

rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 1]

cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=8)
cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=8)
cell_state = cell_fw.zero_state(len(sequence_length), tf.float32)
outputs, state = tf.nn.bidirectional_dynamic_rnn(
  cell_fw=cell_fw,
  cell_bw=cell_bw,
  inputs=rnn_input,
  sequence_length=sequence_length,
  dtype=tf.float32,
  initial_state_fw=cell_state,
  initial_state_bw=cell_state)

outputs_concated = tf.concat(outputs, 2)
state_concated = tf.concat(state, 2)
state_sorted = []
for layer_id in range(num_layers):
  state_sorted.append(state[0][layer_id])
  state_sorted.append(state[1][layer_id])
state_sorted = tuple(state_sorted)
        
def get_last_relevant(outputs_concated, length):
  if not isinstance(length, np.ndarray):
    length = np.array(length)

  batch_size, max_length, hidden_size = outputs_concated.get_shape().as_list()
  index = [max_length * i for i in range(batch_size)] + (length - 1)
  outputs_flat = tf.reshape(outputs_concated, [-1, hidden_size])
  last_relevant = tf.gather(outputs_flat, index)
  return last_relevant

relevant = get_last_relevant(outputs_concated, sequence_length)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_, outputs_concated, state_concated, last_relevant, state_sorted_, state_ = sess.run(
    [outputs, outputs_concated, state_concated, relevant, state_sorted, state], feed_dict=None)

#5.attention test
