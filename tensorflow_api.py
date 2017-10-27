#tf.__version__ >=1.3
import tensorflow as tf
import numpy as np
#One. Matrix operations
#structure:
#1.simple mathematical operation takes an addition as example
#2.tensor replicate
#3 tensor batch replicate

#1.simple mathematical operation takes an addition as example
mat = tf.constant([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
add_row = tf.reduce_sum(mat, 1)
add_col = tf.reduce_sum(mat, 0)
session = tf.Session()
assert (add_row.eval(session=session) == np.array([6, 15, 24, 33])).all()
assert (add_col.eval(session=session) == np.array([22, 26, 30])).all()

#2.tensor replicate
#tile, second parameters 1-D. Length must be the same as the number of dimensions in input
#tf.tile(input, multiples, name=None)
ta = np.array([1, 2, 3])
tb = np.array([[1, 2, 3],[4, 5, 6]])
a_rep = tf.tile(ta, [2])
b_rep = tf.tile(tb, [1, 2])
session = tf.Session()
assert (a_rep.eval(session=session) == np.array([1, 2, 3, 1, 2, 3])).all()
assert (b_rep.eval(session=session) == np.array([[1, 2, 3, 1, 2, 3],[4, 5, 6, 4, 5, 6]])).all()

#tf.fill(dims, value, name=None)
#value is a scalar
f_a = tf.fill([5], 2)
assert (f_a.eval(session=session) == np.array([2, 2, 2, 2, 2])).all()

#3 tensor batch replicate
#The default first dimension is batch, batch in each sample repeated n times
#tile_batch
ta = np.array([[1, 1, 1],[2, 2, 2]])
a_rep = tf.contrib.seq2seq.tile_batch(ta, 3)
session = tf.Session()
print a_rep.eval(session=session)


#Two. Variables and operations
#structure
#1. variable and get_variable

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

#Three. Recurrent Neural Networks(RNN)
#structure:
#1.build LSTM(GRU) cell
#2.build rnn
#3 test single-layer unidirection dynamic_rnn output[:,-1,:] == state
#4.build bidirectional rnn
#5.get dynamic_rnn last time output
#6.get bidirectional dynamic_rnn last time output
#7.multi-layers unidirection rnn and test
#8.multi-layers bidirectional rnn without test
#9.single-layers bidirectional rnn test

#1.build LSTM(GRU) cell
cell = tf.contrib.rnn.BasicLSTMCell(num_units=32, forget_bias=0.0)

#2.build rnn
#dynamic_rnn If there is no initial_state, you must give a dtype.
#as seen in session, feed_dict is not necessary
#dynamic_rnn must have parameters sequence_length, 
#calculated to stop at the length of the sequence
#attention: only test calculated to stop at the length of the sequence
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

#3 test single layer dynamic_rnn output[:,-1,:] == state
#single-layer, single direction rnn pass the test (output[:,-1,:] == state)
rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 1]

cell = tf.contrib.rnn.BasicLSTMCell(num_units=8)
cell_state = cell.zero_state(len(sequence_length), tf.float32)
outputs, state = tf.nn.dynamic_rnn(
  cell=cell,
  inputs=rnn_input,
  sequence_length=sequence_length,
  dtype=tf.float32,
  initial_state=cell_state)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_, state_ = sess.run([outputs, state], feed_dict=None)

#outputs_:
# [[[ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.12647648 -0.11802931  0.09710746 -0.05375184 -0.01846889 -0.07082941
#     0.04427411  0.06775716]
#   [ 0.22429267 -0.25319099  0.20090522 -0.2350363  -0.03346894 -0.15088198
#     0.07085583  0.1138599 ]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]

#  [[ 0.0577014  -0.02319911  0.07561893 -0.52299041 -0.00288081 -0.05581678
#     0.00418655  0.01026991]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]]
#state_:
# LSTMStateTuple(c=array([
#        [ 0.44465992, -0.35971889,  0.3749069 , -0.28752673, -0.22564349,
#         -0.49123293,  0.50713313,  0.15993467],
#        [ 0.09519511, -0.02420649,  0.11231606, -0.58500332, -0.50397503,
#         -0.61437559,  0.86453581,  0.01096342]], dtype=float32),
#         h=array([
#        [ 0.22429267, -0.25319099,  0.20090522, -0.2350363 , -0.03346894,
#         -0.15088198,  0.07085583,  0.1138599 ],
#        [ 0.0577014 , -0.02319911,  0.07561893, -0.52299041, -0.00288081,
#         -0.05581678,  0.00418655,  0.01026991]], dtype=float32))

#4.build bidirectional rnn
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

#5.get dynamic_rnn last time output
#fuction get_last_relevant's return equivalence state,namely last_relevant == state
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

#6.get bidirectional dynamic_rnn last time output
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

#7.multi-layers single-direction rnn and test
#test output[:,-1,:] == state
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

# outputs_:
# [[[ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.00206765  0.0012526  -0.00070076  0.00250549 -0.0010293   0.0020362
#    -0.00075203 -0.00325975]
#   [ 0.00847008  0.00567041 -0.00173253  0.00841379 -0.00419107  0.00634551
#    -0.00162632 -0.01055534]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]

#  [[ 0.00659659  0.0058269  -0.00603103  0.00392975 -0.00379545 -0.00085966
#     0.00274244 -0.00326181]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]]

# state_:
# (LSTMStateTuple(c=array([
#        [-0.38249534, -0.96067655, -0.3418079 , -0.09708116, -0.72390383,
#          0.22623062,  0.29119381, -0.36108243],
#        [-0.57103628, -0.92854673, -0.21461228, -0.07540006, -0.96738023,
#          0.16748942,  0.09926217, -0.77952582]], dtype=float32), 
#       h=array([
#        [-0.16981587, -0.21597899, -0.14614557, -0.0378123 , -0.2779581 ,
#          0.08948451,  0.19159022, -0.20181687],
#        [-0.18480942, -0.03377225, -0.07891702, -0.01703019, -0.26507545,
#          0.03929503,  0.08929662, -0.49588516]], dtype=float32)), 
# LSTMStateTuple(c=array([
#        [-0.10485549,  0.01806218,  0.05592438,  0.13108559,  0.00681598,
#         -0.08692499,  0.03584345,  0.02347218],
#        [-0.01513201,  0.03834464,  0.04773763,  0.12160803,  0.0588663 ,
#         -0.00487377,  0.04261572, -0.06256856]], dtype=float32), 
#       h=array([
#        [-0.05531391,  0.0093281 ,  0.02845607,  0.06787559,  0.00326916,
#         -0.04031532,  0.01762626,  0.01078525],
#        [-0.00785679,  0.01991731,  0.02406042,  0.0641262 ,  0.02748546,
#         -0.00219305,  0.02215303, -0.0278921 ]], dtype=float32)), 
# LSTMStateTuple(c=array([[ 0.01676487,  0.01134046, -0.00343936,  0.01677591, -0.00830541,
#          0.01257986, -0.00328146, -0.02104298],
#        [ 0.0132175 ,  0.01177202, -0.01206709,  0.00780833, -0.0075313 ,
#         -0.00171233,  0.0055482 , -0.0065166 ]], dtype=float32),
#       h=array([[ 0.00847008,  0.00567041, -0.00173253,  0.00841379, -0.00419107,
#          0.00634551, -0.00162632, -0.01055534],
#        [ 0.00659659,  0.0058269 , -0.00603103,  0.00392975, -0.00379545,
#         -0.00085966,  0.00274244, -0.00326181]], dtype=float32)))

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

#8.multi-layers bidirectional rnn without test
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

#9.single-layers bidirectional rnn test
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

#Four. Sequence operations
#structure:
#1 list operation like numpy
#2. reverse sequence
#3 Combines the sequence according to the specified format

#1 list operation like numpy
from tensorflow.python.util import nest
a = nest.map_structure(lambda x: x + 1, [[1,2],[3,4]])
assert a == [[2,3],[4,5]]
#if a is have 3 dims, can't be flatten
a = nest.flatten(((3, 4), 5, (6, 7, (9, 10), 8)))
assert a == [3, 4, 5, 6, 7, 9, 10, 8]

#2. reverse sequence
#tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None, name=None, seq_dim=None,
#batch_dim=None)
#seq_axis, seq_dim equivalence the same as batch_axis, batch_dim
inputs = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 2]
reverse_inputs = tf.reverse_sequence(
  input=rnn_input, 
  seq_lengths=sequence_length, 
  seq_axis=1,
  batch_axis=0)

with tf.Session() as sess:
  reverse_inputs_ = sess.run(reverse_inputs)
#output:
#[[[ 2.  2.  2.]
#  [ 1.  1.  1.]
#  [ 0.  0.  0.]
#  [ 3.  3.  3.]]

# [[ 7.  7.  7.]
#  [ 6.  6.  6.]
#  [ 8.  8.  8.]
#  [ 9.  9.  9.]]]

#3 Combines the sequence according to the specified format
structure = ((3, 4), 5, (6, 7, (9, 10), 8))
flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
packed = nest.pack_sequence_as(structure, flat)
assert packed == (('a', 'b'), 'c', ('d', 'e', ('f', 'g'), 'h'))