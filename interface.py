import os
import sys
import itertools
import sgf
import random
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import functools, operator

def givemelocation(values):
    indes = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18}
    output=[]
    if len(values)==2 and type(values) == str:
        for i in values:
            output.append(indes[i])
        print(output)
    else:
        print("Invalid input value")
    return output

def product(numbers):
    return functools.reduce(operator.mul, numbers)

def init(shape):
    number_inputs_added = product(shape[:-1])
    stddev = 1 / math.sqrt(number_inputs_added)
    return stddev

def fanyi(number):
    a = int(number%19)
    b = int(np.floor(number/19))    
    zidian = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s"]
    c = zidian[a]
    d = zidian[b]
    output = c + d
    return output

def generator(tmp,loc):
    tmp=-tmp
    tmp[:,:,:,-5]=tmp[:,:,:,-4]
    tmp[:,:,:,-4]=tmp[:,:,:,-3]
    tmp[:,:,:,-3]=tmp[:,:,:,-2]
    tmp[:,:,:,-2]=tmp[:,:,:,-1]
    tmp[:,:,:,-1][:,loc] = 1
    return tmp


stddev1 = init([5,5,5,40])
stddev2 = init([3,3,40,40])
stddev3 = init([1,1,40,1])

X = tf.placeholder(tf.float32,shape = [None,19,19,5],name = 'Datasetinput')
Y = tf.placeholder(tf.float32,shape = [None,361],name='Output')

W1 = tf.Variable(tf.truncated_normal([5, 5, 5, 40],stddev = stddev1),name = "W1")
W2 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W2")
W3 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W3")
W4 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W4")
W5 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W5")
W6 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W6")
W7 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W7")
W8 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W8")
W9 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W9")
W10 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W10")
W11 = tf.Variable(tf.truncated_normal([3, 3, 40, 40],stddev = stddev2),name = "W11")
W12 = tf.Variable(tf.truncated_normal([1, 1, 40, 1],stddev = stddev3),name = "W12")
bias = tf.Variable(tf.constant(0, shape=[361], dtype=tf.float32), name="bias")

Z1 = tf.nn.conv2d(X,W1,strides = [1,1,1,1],padding='SAME')
A1 = tf.nn.relu(Z1)
Z2 = tf.nn.conv2d(A1,W2,strides = [1,1,1,1],padding='SAME')
A2 = tf.nn.relu(Z2)
Z3 = tf.nn.conv2d(A2,W3,strides = [1,1,1,1],padding='SAME')
A3 = tf.nn.relu(Z3)
Z4 = tf.nn.conv2d(A3,W4,strides = [1,1,1,1],padding='SAME')
A4 = tf.nn.relu(Z4)
Z5 = tf.nn.conv2d(A4,W5,strides = [1,1,1,1],padding='SAME')
A5 = tf.nn.relu(Z5)
Z6 = tf.nn.conv2d(A5,W6,strides = [1,1,1,1],padding='SAME')
A6 = tf.nn.relu(Z6)
Z7 = tf.nn.conv2d(A6,W7,strides = [1,1,1,1],padding='SAME')
A7 = tf.nn.relu(Z7)
Z8 = tf.nn.conv2d(A7,W8,strides = [1,1,1,1],padding='SAME')
A8 = tf.nn.relu(Z8)
Z9 = tf.nn.conv2d(A8,W9,strides = [1,1,1,1],padding='SAME')
A9 = tf.nn.relu(Z9)
Z10 = tf.nn.conv2d(A9,W10,strides = [1,1,1,1],padding='SAME')
A10 = tf.nn.relu(Z10)
Z11 = tf.nn.conv2d(A10,W11,strides = [1,1,1,1],padding='SAME')
A11 = tf.nn.relu(Z11)
Z12 = tf.nn.conv2d(A11,W12,strides = [1,1,1,1],padding='SAME')
A12 = tf.contrib.layers.flatten(Z12)
output = tf.nn.bias_add(A12,bias,name=None)
tmp = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=output)
prediction = tf.nn.softmax(output)
cost = tf.reduce_mean(tmp)

tmp = tf.argmax(prediction,output_type=tf.int64)
tmp1 = tf.argmax(Y,output_type=tf.int64)

correct_prediction = tf.equal(tmp, tmp1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

input_loc = []
X_tmp = np.zeros([1,19,19,5])
with tf.Session() as sess:
    while not input_loc == "quit":
        if input_loc == "quit":
            break
        input_loc = input("if you want to quit, type in 'quit', or Input your stone's move:")
        X_loc = givemelocation(input_loc)
        X_tmp = generator(X_tmp,X_loc)
        saver.restore(sess,"model.ckpt")
        output = sess.run(tmp, feed_dict={X:X_tmp})
        output1 = sess.run(prediction, feed_dict={X:X_tmp})
        output2 = np.argmax(output1)
        print(fanyi(output2))
	

















