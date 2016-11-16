import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt 
import tensorflow as tf

df = pd.read_csv("data/CHD.csv", header=0)


# Parameters

learning_rate = 0.2
training_epochs = 5
batch_size = 100
display_step = 1
sess = tf.Session()
b=np.zeros((100,2))
#print pd.get_dummies(df['admit']).values[1]
print sess.run(tf.one_hot(indices = [1, 3, 2, 4], depth=5, on_value = 1, off_value = 0, axis = 1 , name = "a"))
#print a.eval(session=sess)


# tf Graph Input

x = tf.placeholder("float", [None, 1]) 
y = tf.placeholder("float", [None, 2]) 
# Create model
# Set model weights
W = tf.Variable(tf.zeros([1, 2]))
b = tf.Variable(tf.zeros([2]))


# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) 
# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) # Cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent


# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph

with tf.Session() as sess:
    tf.train.write_graph(sess.graph, './graphs','graph.pbtxt')
    sess.run(init)
    writer = tf.train.SummaryWriter('./graphs', sess.graph)
    #Initialize the graph structure
    
    graphnumber=321
    
    #Generate a new graph
    plt.figure(1)
    
    #Iterate through all the epochs
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 400/batch_size
        # Loop over all batches

        for i in range(total_batch):
            # Transform the array into a one hot format
            
            temp=tf.one_hot(indices = df['chd'].values, depth=2, on_value = 1, off_value = 0, axis = -1 , name = "a")      
            batch_xs, batch_ys = (np.transpose([df['age']])-44.38)/11.721327, temp
            
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs.astype(float), y: batch_ys.eval()})
            
            # Compute average loss, suming the corrent cost divided by the batch total number
            avg_cost += sess.run(cost, feed_dict={x: batch_xs.astype(float), y: batch_ys.eval()})/total_batch
        # Display logs per epoch step

        if epoch % display_step == 0:
            print "Epoch:", '%05d' % (epoch+1), "cost=", "{:.8f}".format(avg_cost)
            
            #Generate a new graph, and add it to the complete graph
            
            trX = np.linspace(-30, 30, 100) 
            print (b.eval())
            print (W.eval())
            Wdos=2*W.eval()[0][0]/11.721327
            bdos=2*b.eval()[0]
            
            # Generate the probabiliy function
            trY = np.exp(-(Wdos*trX)+bdos)/(1+np.exp(-(Wdos*trX)+bdos) )
            
            # Draw the samples and the probability function, whithout the normalization
            plt.subplot(graphnumber)
            graphnumber=graphnumber+1
            
            #Plot a scatter draw of the random datapoints
            plt.scatter((df['age']),df['chd']) 
            plt.plot(trX+44.38,trY) #Plot a scatter draw of the random datapoints
            plt.grid(True)
            
        #Plot the final graph
        plt.savefig("test.svg")
            

