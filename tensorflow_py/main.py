"""
Phase 1: assemble a graph
Phase 2: use a session to execute operations and evaluate variables in the graph
"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_graph():
    a = tf.constant(10, name='ops_a')
    b = tf.constant(20, name='ops_b')
    c = tf.add(a,b, name = 'ops_c')
    d = tf.constant(value = [[2,4],[6,9]], dtype = tf.int32, name = 'ops_d')
    e = tf.ones_like(d, name = 'ops_e')
    f = tf.add(d,e, name = 'ops_f')
    g = tf.add(f, c, name = 'ops_g')
    h = tf.get_variable(initializer = g, name = 'var_h')
    return f,h

def create_another_graph():
    var_shape = [2,3]
    #var1 = tf.Variable(tf.truncated_normal(var_shape), name = 'var_1')
    var1 = tf.get_variable(initializer = tf.truncated_normal(var_shape), name = 'var_1')
    # Wrong == var2 = tf.Variable(var1*5)
    var2 = tf.Variable(var1.initialized_value()*5, name = 'var_2')
    placeholder = tf.placeholder(dtype = tf.float32, shape = var_shape, name = 'placeholder_1')
    out = tf.add(var2, placeholder, name = 'ops_out')
    return var1,var2,placeholder,out

def main():
    print("****************Tensorflow Tutorials*****************")
    print('Tensorflow-version ' + str(tf.VERSION))
    f,h = create_graph()
    var_1,var_2,holder,out = create_another_graph()
    assign_op = tf.assign(ref = h, value = f, validate_shape = True)
    graph = tf.get_default_graph()
    #print(dir(graph))
    #print(graph.get_operations())
    #print([ops.values() for ops in graph.get_operations()])
    c_ = graph.get_operation_by_name("ops_g").outputs[0]
    c__ = graph.get_tensor_by_name('ops_g:0')
    #print(graph.as_graph_def())
    writer = tf.summary.FileWriter('./graphs', graph)
    with tf.Session() as sess:
        print(sess.run(tf.report_uninitialized_variables()))
        #sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer([h]))
        sess.run(h.initializer)
        arr = sess.run(c__)
        #print(h.eval())
        print(sess.run(sess.graph.get_tensor_by_name('var_h:0')))
        #q = sess.graph.get_tensor_by_name('var_h:0')
        sess.run(assign_op)
        print(h.eval())
        #print(q.eval())
        print('****Exampele2*****')
        sess.run(tf.variables_initializer([var_1, var_2]))
        print(var_2.eval())

        ops_list = graph.get_operations()
        tensor_list = np.array([ops.values() for ops in ops_list])
        feedable = np.array([graph.is_feedable(tensor) for tensor in tensor_list])
        feed = list(zip(tensor_list,feedable))
        print(feed)

        print(sess.run(out, feed_dict = {holder: np.array([[1.,2.,3.],[1.,2.,3.]], dtype = np.float32)}))
        print('*****End of examples******')

    print(arr.shape) # This is a numpy array
    writer.close()

    print("***************End of the program**********************")

if __name__ == '__main__':
  main()
