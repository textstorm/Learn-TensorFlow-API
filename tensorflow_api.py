
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
#1.