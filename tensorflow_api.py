
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