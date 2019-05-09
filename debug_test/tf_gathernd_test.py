import tensorflow as tf
import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b) + tf.constant(1e-10)
    c = tf.cast(c, dtype=tf.float64)
    c = c + tf.constant(1e-10, dtype=tf.float64)

with tf.Session() as sess:
    print (sess.run(c))