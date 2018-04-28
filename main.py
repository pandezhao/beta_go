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


data_dirs = "sum"
model_path = "./model.ckpt"
output_class = 361
learning_rate = 0.0001
epochs = 80
mini_batch_size = 400


def load_sgf(*dataset_dirs): 
    for dataset_dir in dataset_dirs:
        full_dir = os.path.join(os.getcwd(), dataset_dir)
        dataset_files = [os.path.join(full_dir, name) for name in os.listdir(full_dir)]
        for f in dataset_files:
            if os.path.isfile(f) and f.endswith(".sgf"):
                yield f


def get_data_sets(*data_sets):
    sgf_files = list(load_sgf(*data_sets))
    print("%s sgfs is found." % len(sgf_files))
    return sgf_files


def product(numbers):
    return functools.reduce(operator.mul, numbers)

def init(shape):
    number_inputs_added = product(shape[:-1])
    stddev = 1 / math.sqrt(number_inputs_added)
    return stddev

def gamereplay(node):
    chunk = np.zeros([19,19,1])
    while node is not None:
        tmp = chunk[:,:,-1]
        tmp = translate(tmp,node.properties)
        chunk = np.concatenate((chunk,tmp),axis=2)
        node = node.next
    return chunk


def translate(board,properties):
    if str(properties.keys()) == "dict_keys(['B'])":
        a,b=givemelocation(properties['B'])
        board[a][b] = 1
    if str(properties.keys()) == "dict_keys(['W'])":
        a,b=givemelocation(properties['W'])
        board[a][b] = -1
    return board.reshape(19,19,1)


def givemelocation(values):
    indes = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18}
    if values[0]:
        a = indes[str(values[0][0])]
        b = indes[str(values[0][1])]
    else:
        a = None
        b = None
    return a,b


def wishdata(samplelist):
    samples = []
    print("There are about "+str(len(samplelist)*10)+" samples to go")
    for i in range(len(samplelist)):
        tmp = samplelist[i]
        number_of_this_game = np.floor((tmp.shape[2]-6)/10)
        if number_of_this_game < 5:
            continue
        ran = random.sample(range(tmp.shape[2]-6),10)
        for j in range(len(ran)):
            sample={}
            y = np.zeros((361))
            tmp_tmp = tmp[:,:,ran[j]:ran[j]+5]
            color = np.sum(tmp[:,:,ran[j]+6]-tmp[:,:,ran[j]+5])
            z = np.where(tmp[:,:,ran[j]+6]-tmp[:,:,ran[j]+5] == color)
            y[z[0]*19+z[1]] = color
            sample['x']=tmp_tmp
            sample['y']=y
            samples.append(sample)
        if i % 1000 == 0:
            print(str(10*i)+" training set has been generated")
    return samples


def generate_training_set(sgf_files):
    samples = []
    for i in range(len(sgf_files)): 
        t = sgf_files[i]
        h = open(t)
        collection = sgf.parse(h.read())
        game=collection.children[0]
        node = game.root
        a = gamereplay(node)
        samples.append(a)
        if i % 1000 == 0:
            print(str(i)+" sgf files has been read")
    precessed_data = wishdata(samples)
    print("%s training set has been generated" % len(precessed_data))
    return precessed_data


def preprocess(*data_sets, processed_dir="processed_data"):
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)
    sgf_files = get_data_sets(*data_sets)
    data_set = generate_training_set(sgf_files)
    return data_set


data_set = preprocess(data_dirs, processed_dir="processed_data")

def random_mini_batches(X, Y, mini_batch_size = 64, epoches = 1):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = len(X)                  
    mini_batches = []
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    tmp = list(zip(X,Y))
    for i in range(epoches):
        random.shuffle(tmp)
        X[:],Y[:] = zip(*tmp)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = X[k * mini_batch_size : (k+1) * mini_batch_size]
            mini_batch_Y = Y[k * mini_batch_size : (k+1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch_X = X[num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = Y[num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    return mini_batches

# divide the dataset into training set and test set
X_train = []
Y_train = []
X_test = []
Y_test = []
print("doing a little modification on training set, it won't take long")
for i in range(int(np.floor(0.98 * len(data_set)))):
    if np.sum(data_set[i]['y'])==1:
        X_train.append(data_set[i]['x'])
        Y_train.append(data_set[i]['y'])
    if np.sum(data_set[i]['y'])== -1:
        X_train.append(data_set[i]['x']*-1)
        Y_train.append(data_set[i]['y']*-1)
print("doing a little modification on test set, it won't take long")
for i in range(int(np.floor(0.98 * len(data_set))),len(data_set)):
    if np.sum(data_set[i]['y'])==1:
        X_test.append(data_set[i]['x'])
        Y_test.append(data_set[i]['y'])
    if np.sum(data_set[i]['y'])== -1:
        X_test.append(data_set[i]['x']*-1)
        Y_test.append(data_set[i]['y']*-1)
print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))

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
print(Z1)
A1 = tf.nn.relu(Z1)
print(A1)
Z2 = tf.nn.conv2d(A1,W2,strides = [1,1,1,1],padding='SAME')
print(Z2)
A2 = tf.nn.relu(Z2)
print(A2)
Z3 = tf.nn.conv2d(A2,W3,strides = [1,1,1,1],padding='SAME')
print(Z3)
A3 = tf.nn.relu(Z3)
print(A3)
Z4 = tf.nn.conv2d(A3,W4,strides = [1,1,1,1],padding='SAME')
print(Z4)
A4 = tf.nn.relu(Z4)
print(A4)
Z5 = tf.nn.conv2d(A4,W5,strides = [1,1,1,1],padding='SAME')
print(Z5)
A5 = tf.nn.relu(Z5)
print(A5)
Z6 = tf.nn.conv2d(A5,W6,strides = [1,1,1,1],padding='SAME')
print(Z6)
A6 = tf.nn.relu(Z6)
print(A6)
Z7 = tf.nn.conv2d(A6,W7,strides = [1,1,1,1],padding='SAME')
print(Z7)
A7 = tf.nn.relu(Z7)
print(A7)
Z8 = tf.nn.conv2d(A7,W8,strides = [1,1,1,1],padding='SAME')
print(Z7)
A8 = tf.nn.relu(Z8)
print(A8)
Z9 = tf.nn.conv2d(A8,W9,strides = [1,1,1,1],padding='SAME')
print(Z9)
A9 = tf.nn.relu(Z9)
print(A9)
Z10 = tf.nn.conv2d(A9,W10,strides = [1,1,1,1],padding='SAME')
print(Z10)
A10 = tf.nn.relu(Z10)
print(A10)
Z11 = tf.nn.conv2d(A10,W11,strides = [1,1,1,1],padding='SAME')
print(Z11)
A11 = tf.nn.relu(Z11)
print(A11)
Z12 = tf.nn.conv2d(A11,W12,strides = [1,1,1,1],padding='SAME')
print(Z12)
A12 = tf.contrib.layers.flatten(Z12)
print(A12)
output = tf.nn.bias_add(A12,bias,name=None)
print(output)
tmp = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=output)
print(tmp)
prediction = tf.nn.softmax(output)
print(prediction)
cost = tf.reduce_mean(tmp)
print(cost)

tmp = tf.argmax(prediction,output_type=tf.int64)
tmp1 = tf.argmax(Y,output_type=tf.int64)

correct_prediction = tf.equal(tmp, tmp1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

Test_cost = []
Train_cost = []
Train_Accuracy = []
Test_Accuracy = []
counter = 0

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        minibatches = random_mini_batches(X_train,Y_train, mini_batch_size,1)
        for minibatch in minibatches:
            counter += 1
            (minibatch_X, minibatch_Y) = minibatch 
            _ , temp_cost , train_accuracy = sess.run([optimizer,cost,accuracy],feed_dict={X:minibatch_X,Y:minibatch_Y})
            Train_cost.append(temp_cost)
            Train_Accuracy.append(train_accuracy)

            test_cost, test_accuracy = sess.run([cost,accuracy],feed_dict={X:X_test,Y:Y_test})
            Test_cost.append(test_cost)
            Test_Accuracy.append(test_accuracy)
            if counter % 10 == 0:
                print("Cost after %i iterations training is: %f, epochs is: %i" % (counter, temp_cost, i))
                print("Train accuracy: %f Test accuracy: %f" % (train_accuracy, test_accuracy))
            
    
    plt.plot(np.squeeze(Train_cost))
    plt.ylabel('train cost')
    plt.xlabel('number of iterations')
    plt.title("Learning curve when learning rate = "+str(learning_rate))
    plt.savefig("Train cost.jpg")    
    plt.show()

    ## 下面是作圖專用
    plt.plot(np.squeeze(Test_cost))
    plt.ylabel('test cost')
    plt.xlabel('number of iterations')
    plt.title("Accuracy changes for learning rate = "+str(learning_rate))
    plt.savefig("Test cost.jpg")    
    plt.show()

    ##
    plt.plot(np.squeeze(Test_cost))
    plt.plot(np.squeeze(Train_cost))
    plt.ylabel('comparsion between training cost and test cost')
    plt.xlabel('number of iterations')
    plt.title("Accuracy changes for learning rate = "+str(learning_rate))
    plt.savefig("Train cost and test cost.jpg")    
    plt.show()


   
    ##
    plt.plot(np.squeeze(Train_Accuracy))
    plt.ylabel('Training set Accuracy')
    plt.xlabel('number of iterations')
    plt.title("Accuracy changes for learning rate = "+str(learning_rate))
    plt.savefig("Train accuracy.jpg")
    plt.show()
    ##
    plt.plot(np.squeeze(Test_Accuracy))
    plt.ylabel('Test set Accuracy')
    plt.xlabel('number of iterations')
    plt.title("Accuracy changes for learning rate = "+str(learning_rate))
    plt.savefig("Test accuracy.jpg")
    plt.show()

    saver.save(sess, model_path)
    

def locationmegive(output):
    output=output.reshape(19,19)
    locat = np.where(output !=0)
    zidian = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s"]
    print(locat)
    move1 = zidian[int(locat[0])]
    move2 = zidian[int(locat[1])]
    move = move1 + move2
    return move






















