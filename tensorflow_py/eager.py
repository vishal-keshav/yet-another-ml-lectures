import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

if __name__ == '__main__':
    tf.enable_eager_execution()

    if tf.executing_eagerly():
        print('Eager execution is enables')

    #Graph in eager
    a = tf.constant(10)
    print(a)
    b = tf.constant(20)
    print(b)
    if tf.equal(b,20):
        print('A tensor can be compared aswell')
    c = tf.add(a,b)
    print(c)
    d = tf.constant(value = [[2,4],[6,9]])
    print(d)
    print(d.numpy())
    e = tf.ones_like(d)
    print(e)
    f = tf.add(d,e)
    print(f)
    g = tf.add(f, c)
    print(g)
    h = tfe.Variable(g)
    print(h.numpy())
    h.assign(d)
    print(h.numpy())
