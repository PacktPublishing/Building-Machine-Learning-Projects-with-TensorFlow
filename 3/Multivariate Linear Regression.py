import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

df = pd.read_csv("data/boston.csv", header=0)
print df.describe()

f, ax1 = plt.subplots()
plt.figure() # Create a new figure

y = df['MEDV']

for i in range (1,8):
    number = 420 + i
    ax1.locator_params(nbins=3)
    ax1 = plt.subplot(number)
    plt.title(list(df)[i])
    ax1.scatter(df[df.columns[i]],y) #Plot a scatter draw of the  datapoints
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


X = tf.placeholder("float", name="X") # create symbolic variables
Y = tf.placeholder("float", name = "Y")


with tf.name_scope("Model"):

    w = tf.Variable(tf.random_normal([2], stddev=0.01), name="b0") # create a shared variable
    b = tf.Variable(tf.random_normal([2], stddev=0.01), name="b1") # create a shared variable
    
    def model(X, w, b):
        return tf.mul(X, w) + b # We just define the line as X*w + b0  

    y_model = model(X, w, b)

with tf.name_scope("CostFunction"):
    cost = tf.reduce_mean(tf.pow(Y-y_model, 2)) # use sqr error for cost function

train_op = tf.train.AdamOptimizer(0.001).minimize(cost)


sess = tf.Session()
init = tf.initialize_all_variables()
tf.train.write_graph(sess.graph, '/home/bonnin/linear2','graph.pbtxt')
cost_op = tf.scalar_summary("loss", cost)
merged = tf.merge_all_summaries()
sess.run(init)
writer = tf.train.SummaryWriter('/home/bonnin/linear2', sess.graph)

xvalues = df[[df.columns[2], df.columns[4]]].values.astype(float)
yvalues = df[df.columns[12]].values.astype(float)
b0temp=b.eval(session=sess)
b1temp=w.eval(session=sess)


for a in range (1,50):
    cost1=0.0
    for i, j in zip(xvalues, yvalues):   
        sess.run(train_op, feed_dict={X: i, Y: j}) 
        cost1+=sess.run(cost, feed_dict={X: i, Y: i})/506.00
        #writer.add_summary(summary_str, i) 
    xvalues, yvalues = shuffle (xvalues, yvalues)
    print (cost1)
    b0temp=b.eval(session=sess)
    b1temp=w.eval(session=sess)
    print (b0temp)
    print (b1temp)

