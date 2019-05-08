import tensorflow as tf

batch_size = 8
dag_hidden_dim = 100
idx_var=tf.constant(0) #tf.Variable(0,trainable=False)

representation = tf.zeros([batch_size, 1, dag_hidden_dim])
temp = tf.placeholder(tf.int32, [None, None, None])
positions = tf.gather(temp, idx_var, axis=1)

batch_size = tf.shape(positions)[0]
neigh_num = tf.shape(positions)[1]

idxs = tf.range(0, limit=batch_size) # [batch]
idxs = tf.reshape(idxs, [-1, 1]) # [batch_size, 1]
idxs = tf.tile(idxs, [1, neigh_num]) # [batch_size, neigh_num]
indices = tf.stack((idxs,positions), axis=2) # [batch_size, neigh_num, 2]
    
# your code here
print representation.get_shape()
print indices.get_shape()

result = tf.gather_nd(representation, indices)
print result.get_shape()