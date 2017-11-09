#tf.__version__ >=1.3
#python 2.7
import tensorflow as tf
import numpy as np
#One. Matrix operations
#structure:
#1.simple mathematical operation takes an addition as example
#2.tensor replicate
#3.tensor batch replicate
#4.other operation

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

#slice matrix into a list of vectors at once
#slice operation is very inefficient
sess = tf.Session()
x = tf.random_uniform([4, 3])
for i in tf.unstack(x):
  print sess.run(i)

#4 other operation
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
with tf.Session() as sess:
  print sess.run(tf.stack([x, y, z]))
  print sess.run(tf.stack(x))         #[1, 4]
  print sess.run(tf.stack([x]))       #[[1, 4]]
# [[1 4]
#  [2 5]
#  [3 6]]

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

#3 test single-layer unidirection dynamic_rnn output[:,-1,:] == state
#single-layer, single direction rnn pass the test (output[:,-1,:] == state)
rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 2]

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
#   [ 0.00153795  0.07732481 -0.08294688 -0.11574274  0.0890636  -0.03697445
#     0.07978623  0.008425  ]
#   [ 0.00174299  0.18283756 -0.17022766 -0.32612532  0.11344922 -0.10089301
#     0.14200805  0.01964835]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]

#  [[ 0.01643418  0.0552044  -0.07828833 -0.45551524  0.00513759 -0.18611874
#     0.03794847  0.00502565]
#   [ 0.07937566  0.07880038 -0.07086038 -0.63650453  0.00499556 -0.30451366
#     0.03089749  0.00476533]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]]

#state_:
# LSTMStateTuple(c=array([
#         [ 0.00323261,  0.25609112, -0.32218587, -0.66404641,  0.54425246,
#         -0.18864585,  0.56962621,  0.06972988],
#         [ 0.13235423,  0.08185608, -0.11163406, -1.85610366,  0.33016002,
#         -0.5184828 ,  1.09495151,  0.14088134]], dtype=float32), 
#         h=array([
#         [ 0.00174299,  0.18283756, -0.17022766, -0.32612532,  0.11344922,
#          -0.10089301,  0.14200805,  0.01964835],
#         [ 0.07937566,  0.07880038, -0.07086038, -0.63650453,  0.00499556,
#          -0.30451366,  0.03089749,  0.00476533]], dtype=float32))

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

#7.multi-layers unidirection rnn and test
#test output[:,-1,:] == state
import tensorflow as tf
import numpy as np
rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 2]

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
#   [-0.00068807  0.0008918  -0.00022857  0.00108959 -0.00044004  0.00062247
#    -0.00147907  0.00078881]
#   [-0.00296071  0.00232844 -0.00218854  0.00426432 -0.00234694  0.00267158
#    -0.00431311  0.00249919]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]

#  [[-0.00378048 -0.00099598 -0.0050701   0.0050314  -0.00555934  0.00301681
#     0.00043598 -0.00017459]
#   [-0.01115858 -0.00337714 -0.01449249  0.01590648 -0.01867254  0.00511031
#    -0.00047708 -0.00038018]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]
#   [ 0.          0.          0.          0.          0.          0.          0.
#     0.        ]]]

# state_:
# (LSTMStateTuple(c=array([
#        [-0.10900721,  0.44434682,  0.14086804, -0.03247508, -0.26559615,
#          0.22433957, -0.14278848, -0.77595639],
#        [-0.0241932 ,  1.11304605,  0.20688078, -0.18489121, -0.41910362,
#          0.87090784, -0.65774769, -1.78042471]], dtype=float32), 
#        h=array([
#        [-0.05708936,  0.23178022,  0.07186922, -0.01970752, -0.07174787,
#          0.12926365, -0.08449236, -0.15205833],
#        [-0.01413995,  0.56629145,  0.10679603, -0.14572874, -0.01304164,
#          0.5571593 , -0.46583822, -0.01527309]], dtype=float32)), 
# LSTMStateTuple(c=array([
#        [ 0.0039802 ,  0.03470499, -0.05144119, -0.06089086, -0.05795141,
#          0.03572204, -0.05287341, -0.03314351],
#        [-0.19515125,  0.10581306, -0.26675332, -0.14536369, -0.07693268,
#          0.12149764, -0.17650022, -0.17372355]], dtype=float32), 
#        h=array([
#        [ 0.00195208,  0.01742339, -0.02540538, -0.02992798, -0.02893698,
#          0.01881238, -0.02738174, -0.01651184],
#        [-0.1012024 ,  0.04516158, -0.11642321, -0.06861219, -0.03982964,
#          0.06868713, -0.09045848, -0.08286985]], dtype=float32)), 
# LSTMStateTuple(c=array([
#        [-0.00589246,  0.00468168, -0.00438315,  0.00848721, -0.00465612,
#          0.0053333 , -0.00868384,  0.00505709],
#        [-0.0223921 , -0.00684667, -0.02969529,  0.03169667, -0.03559118,
#          0.01014503, -0.00095627, -0.00077794]], dtype=float32), 
#        h=array([
#        [-0.00296071,  0.00232844, -0.00218854,  0.00426432, -0.00234694,
#          0.00267158, -0.00431311,  0.00249919],
#        [-0.01115858, -0.00337714, -0.01449249,  0.01590648, -0.01867254,
#          0.00511031, -0.00047708, -0.00038018]], dtype=float32)))

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
#attention: in forward, the last state is last time-step, in backward, the 
#last state is first time-step, outputs_concated pass tests
import tensorflow as tf
import numpy as np
num_layers = 2

rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 2]

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

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_, state_, outputs_concated_, = sess.run(
    [outputs, state, outputs_concated], feed_dict=None)

# outputs_:
# (array([[[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ,  0.        ],
#         [ 0.04048752,  0.12491627,  0.03836499, -0.07948427, -0.04160167,
#           0.05925818,  0.06162658, -0.02842409],
#         [ 0.08196551,  0.23916924,  0.14841855, -0.15591714, -0.07177892,
#           0.10414544,  0.18179366, -0.05716891],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ,  0.        ]],

#        [[ 0.04348128,  0.12894364,  0.44175708, -0.03187086, -0.00897511,
#           0.00438761,  0.51877367, -0.00672479],
#         [ 0.04784355,  0.17153871,  0.61990917, -0.04831723, -0.00577151,
#           0.00615267,  0.72132194, -0.0094572 ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ,  0.        ]]], dtype=float32), 
#  array([[[-0.19642846,  0.06373812,  0.08301786,  0.10088141,  0.17812994,
#          -0.05318081,  0.00688785,  0.20924002],
#         [-0.14067782,  0.03135044,  0.14990938,  0.16472881,  0.23402362,
#          -0.06020171,  0.02405675,  0.22662038],
#         [-0.05053597,  0.01084952,  0.12755617,  0.14480837,  0.14599265,
#          -0.04374506,  0.01940634,  0.16975389],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ,  0.        ]],

#        [[-0.00277134,  0.00134958,  0.3600471 ,  0.50303167,  0.07941235,
#          -0.06925385,  0.00635141,  0.13354544],
#         [-0.00072066,  0.0005523 ,  0.17546897,  0.36836356,  0.03531501,
#          -0.04667888,  0.00395748,  0.09314852],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#           0.        ,  0.        ,  0.        ]]], dtype=float32))

#state_:
# (LSTMStateTuple(c=array([
#        [ 0.23735575,  0.53883046,  0.19414832, -0.35065901, -0.14806649,
#          0.22898148,  0.28465283, -0.16816521],
#        [ 0.4350251 ,  0.40298128,  0.73862982, -0.13515387, -0.0125941 ,
#          0.01704465,  1.11337817, -0.08421063]], dtype=float32), 
#        h=array([
#        [ 0.08196551,  0.23916924,  0.14841855, -0.15591714, -0.07177892,
#          0.10414544,  0.18179366, -0.05716891],
#        [ 0.04784355,  0.17153871,  0.61990917, -0.04831723, -0.00577151,
#          0.00615267,  0.72132194, -0.0094572 ]], dtype=float32)), 
# LSTMStateTuple(c=array([
#        [-0.40336657,  0.13885151,  0.16273995,  0.20037238,  0.39734024,
#         -0.09810479,  0.01418078,  0.46064973],
#        [-0.75206769,  0.03766214,  0.39694417,  1.62635589,  0.22310354,
#         -0.10648929,  0.20933509,  1.36511719]], dtype=float32), 
#        h=array([
#        [-0.19642846,  0.06373812,  0.08301786,  0.10088141,  0.17812994,
#         -0.05318081,  0.00688785,  0.20924002],
#        [-0.00277134,  0.00134958,  0.3600471 ,  0.50303167,  0.07941235,
#         -0.06925385,  0.00635141,  0.13354544]], dtype=float32)))
# outputs_concated:
# [[[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#     -1.96428463e-01   6.37381151e-02   8.30178633e-02   1.00881413e-01
#      1.78129941e-01  -5.31808063e-02   6.88784616e-03   2.09240019e-01]
#   [  4.04875204e-02   1.24916270e-01   3.83649878e-02  -7.94842690e-02
#     -4.16016690e-02   5.92581816e-02   6.16265833e-02  -2.84240935e-02
#     -1.40677825e-01   3.13504376e-02   1.49909377e-01   1.64728805e-01
#      2.34023616e-01  -6.02017082e-02   2.40567494e-02   2.26620376e-01]
#   [  8.19655135e-02   2.39169240e-01   1.48418546e-01  -1.55917138e-01
#     -7.17789233e-02   1.04145445e-01   1.81793660e-01  -5.71689121e-02
#     -5.05359694e-02   1.08495215e-02   1.27556175e-01   1.44808367e-01
#      1.45992652e-01  -4.37450632e-02   1.94063354e-02   1.69753894e-01]
#   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]]

#  [[  4.34812754e-02   1.28943637e-01   4.41757083e-01  -3.18708606e-02
#     -8.97510722e-03   4.38760780e-03   5.18773675e-01  -6.72478648e-03
#     -2.77133798e-03   1.34957582e-03   3.60047102e-01   5.03031671e-01
#      7.94123486e-02  -6.92538545e-02   6.35140669e-03   1.33545443e-01]
#   [  4.78435457e-02   1.71538711e-01   6.19909167e-01  -4.83172312e-02
#     -5.77150844e-03   6.15267223e-03   7.21321940e-01  -9.45719518e-03
#     -7.20655604e-04   5.52301528e-04   1.75468966e-01   3.68363559e-01
#      3.53150107e-02  -4.66788784e-02   3.95748066e-03   9.31485221e-02]
#   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]
#   [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#      0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00]]]

#10.multi-layers bidirectional rnn test
import tensorflow as tf
import numpy as np
num_layers = 4
num_bi_layers = num_layers / 2

rnn_input = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
sequence_length = [3, 2]

def build_rnn_cell(num_units, num_layers):
  cell_list = []
  for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(num_units)
    cell_list.append(cell)
  if num_layers == 1:
    return cell_list[0]
  else:
    return tf.contrib.rnn.MultiRNNCell(cell_list)

cell_fw = build_rnn_cell(num_units=8, num_layers=num_bi_layers)
cell_bw = build_rnn_cell(num_units=8, num_layers=num_bi_layers)
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
state_sorted = []
for layer_id in range(num_bi_layers):
  state_sorted.append(state[0][layer_id])
  state_sorted.append(state[1][layer_id])
state_sorted = tuple(state_sorted)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outputs_, state_, outputs_concated_, state_sorted_ = sess.run(
    [outputs, state, outputs_concated, state_sorted], feed_dict=None)

#raw state orgnazation structure:
#(fw/bw, layers, c/h)
#((LSTMTuple(), LSTMTuple()), (LSTMTuple(), LSTMTuple()))
#((LSTMStateTuple(c=<>, h=<>),  fw_1
#  LSTMStateTuple(c=<>, h=<>)), fw_2
# (LSTMStateTuple(c=<>, h=<>),  bw_1
#  LSTMStateTuple(c=<>, h=<>))) bw_2

#outputs orgnazation structure:
#(output_fw, output_bw)

#11.attention test
#simply implement the attention without using the wrapper
#
outputs = np.asarray([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                    [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]]],
                    dtype=np.float32)
weight = np.asarray([[2, 2, 2], [2, 2, 2]], dtype=np.float32)

#attention = tf.matmul(outputs, weight)
#use the above statement directly, error: Shape must be rank 2 but is rank 3 for 
#'MatMul_1' (op: 'MatMul') with input shapes: [2,4,3], [2,3].
weight_expand = tf.expand_dims(weight, 1)
tmp = weight_expand * outputs
attention = tf.reduce_sum(tmp, 2)

with tf.Session() as sess:
    print sess.run(attention)

# [[  0.   6.  12.  18.]
#  [ 36.  42.  48.  54.]]

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

#Other
#1. loop
#tf.while_loop(cond, body, loop_vars)
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])
session = tf.Session()
assert r.eval(session=session) == 10

#gradient calculation
#1.simple calculate gradient
#the object of the gradient calculation can be a constant
x = tf.constant([1, 2, 3, 4], dtype=tf.float32)
y = x ** 2
g = tf.gradients(y, x)
session = tf.Session()
print session.run(g)
#[array([ 2.,  4.,  6.,  8.], dtype=float32)]

#2.do other operation on the gradient
#the object of the gradient calculation is variables
#if variables are present must use tf.global_variables_initilizer
with tf.Graph().as_default():
  x = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
  y = x ** 2
  tvar = tf.trainable_variables()
  g = tf.gradients(y, tvar)
  g_cliped, global_norm = tf.clip_by_global_norm(g, clip_norm=3.0)
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  print session.run(g_cliped)
#[array([ 2.,  4.,  6.,  8.], dtype=float32)]

#3.Apply the gradient
#this method can use tf.clip_by_global_norm
with tf.Graph().as_default():
  x = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
  y = x ** 2
  tvar = tf.trainable_variables()
  g = tf.gradients(y, tvar)
  g_cliped, global_norm = tf.clip_by_global_norm(g, clip_norm=3.0)
  opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
  update = opt.apply_gradients(zip(g_cliped, tvar))
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  g_cliped_, update_ = session.run([g_cliped, update])
  x_ = session.run(x)
  print g_cliped_
  print x_

#g_cliped_: [array([ 0.54772258,  1.09544516,  1.64316773,  2.19089031], dtype=float32)]
#x_: [ 0.45227742  0.90455484  1.35683227  1.80910969]

#Apply the gradient the second method

with tf.Graph().as_default():
  x = tf.Variable([1, 2, 3, 4], dtype=tf.float32)
  y = x ** 2
  opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
  grads_and_vars = opt.compute_gradients(y)
  #grads_and_vars = [(tf.clip_by_global_norm(g, clip_norm=3.0), v), for g, v in grads_and_vars]
  grads_and_vars = [(tf.clip_by_norm(g, clip_norm=3.0), v) for g, v in grads_and_vars]
  update = opt.apply_gradients(grads_and_vars)
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  grads_and_vars_, _ = session.run([grads_and_vars, update])
  x_ = session.run(x)
  print grads_and_vars_
  print x_

# [(array([ 0.54772258,  1.09544516,  1.64316761,  2.19089031], dtype=float32), 
#   array([ 1.,2.,3.,4.], dtype=float32))]
# [ 0.45227742  0.90455484  1.35683239  1.80910969]

#attention:
#with respect to clip_by_norm and clip_by_global_norm, both can be used to clip the gradients,
#use as above
#but in rnn you had better use clip_by_global_norm
#both of the two clip method are used, sketch rnn use clip_by_norm, nmt use clip_by_global_norm

clip_norm = tf.clip_by_norm(tf.constant([-2, 3, 6], dtype=tf.float32), 5.0)
clip_global_norm = tf.clip_by_global_norm(
  [tf.constant([-2, 3, 6], dtype=tf.float32),
  tf.constant([-4, 6, 12], dtype=tf.float32)], 14.5)
session = tf.Session()
clip_norm_, clip_global_norm_ = session.run([clip_norm, clip_global_norm])

# clip_norm_: clip_norm = 5.0, L2 norm = 7.0
# [-1.42857146  2.14285731  4.28571463] = [-2, 3, 6] * 5 / 7
# clip_global_norm_: clip_norm = 14.5 L2 norm = 7.0, 14.0
# ([array([-1.85274196, 2.77911282, 5.55822563], dtype=float32), 
#   array([-3.70548391, 5.55822563, 11.11645126],dtype=float32)], 
#   global_norm, 15.652476) 
# clip_global_norm_ = [-2, 3, 6] * 14.5 / 15.65, [-4, 6, 12] * 14.5 / 15.65
# global_norm = sqrt(7^2 + 14^2) = 15.652476

#Five.data feeding
#approach 1
#approach 2
#approach 3

#approach 1
#this approach can be very efficient, but isn't flexible.if you use your model with
#another dataset you have to rewrite the graph.
#also,you have to load all the data at once and keep in memory, in sever it may be ok,
#but it word oom in your PC
actual_data = np.ones(10).astype("int32")
data = tf.constant(actual_data)
result = data * 2
with tf.Session() as sess:
  print sess.run(result)

#approach 2
#you can read data from disk, solved the problems in above mothod
#now i use placeholder and load data in memory, ...
data = tf.placeholder(tf.float32)
prediction = tf.square(data) + 1
actual_data = np.random.normal(size=[100])
tf.Session().run(prediction, feed_dict={data: actual_data})

#approach 3
#next chapter data

#six. tf.data
src_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(src_data))
tgt_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(tgt_data))
src_tgt_vocab = lookup_ops.index_table_from_tensor(tf.constant(index2vocab), default_value=0)

#Seven. tf.contrib.lookup
#1.tf.contrib.lookup.index_table_from_tensor
#2.tf.contrib.lookup.index_table_from_file

#1.tf.contrib.lookup.index_table_from_tensor
import random
index_ = random.sample(range(100), 10)
vocab_table = tf.cast(tf.contrib.lookup.index_table_from_tensor(tf.constant(index2word)), tf.int32)
vocab_id = vocab_table.lookup(tf.constant(index2word[index_]))
table_initializer = tf.tables_initializer()
with tf.Session() as session:
  session.run(table_initializer)
  print session.run(vocab_id)

#2.tf.contrib.lookup.index_table_from_file
def vocab_write(index2word):
  f = open("vocab.txt", 'w')
  for word in index2word:
    f.write(word + "\n")
  f.close()

import random
index_ = random.sample(range(100), 10)
vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file="vocab.txt")
vocab_id = vocab_table.lookup(tf.constant(index2word[index_]))
table_initializer = tf.tables_initializer()
with tf.Session() as session:
  session.run(table_initializer)
  print session.run(vocab_id)

#Eight. conditon and loop

#1.condition
#if condition is true return x, else return y
#tf.where(condition, x=None, y=None, name=None)
#tf.cond(pred, true_fn=None, false_fn=None, strict=False, name=None, fn1=None, fn2=None)
#compare with tf.where, tf.cond x,y is func(callable)
tmp = tf.where(5>7, "True", "Flase")

a = tf.constant(5)
b = tf.constant(7)
x = tf.constant(2)
y = tf.constant(5)
z = tf.multiply(a, b)
tmp = tf.cond(tf.cast(x<y, tf.bool), lambda: tf.add(x, z), lambda: tf.square(y))