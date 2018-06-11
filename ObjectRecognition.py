# import necessary libraries
import tensorflow as tf

import numpy as np
import datetime
import matplotlib.pyplot as plt
# import necessary functions
from utilities import *

# load the data
maybe_download_and_extract()
train_,tr_target,_,_ = load_CIFAR10_data()

#view a few examples
img_idx = np.random.choice(40000, 9, replace=False)
sample = train_[img_idx].reshape(9, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
fig, axes1 = plt.subplots(3,3,figsize=(3,3))
i=0
for j in range(3):
    for k in range(3):
        #i = np.random.choice(range(len(sample)),replace=False)
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(sample[i])
        i=i+1
plt.show()

# Declare the necessary values which will be used for training.
input_size = 32*32*3
hidden_size = 75
batch_size = 200
learning_rate = 7.5e-4
parameters = {}

### Add names to values below ###
x = tf.placeholder(tf.float32, shape=(None,input_size),name = "X")
y = tf.placeholder(tf.float32,shape=(None,10),name = "Y")
parameters['w1'] = tf.Variable(tf.random_normal([input_size,hidden_size]),name = 'w1')
parameters['b1'] = tf.Variable(tf.random_normal([hidden_size]),name = 'b1')
parameters['w2'] = tf.Variable(tf.random_normal([hidden_size,10]),name = 'w2')
parameters['b2'] = tf.Variable(tf.random_normal([10]),name='b2')
#### End of naming values ####    

### generate histograms for weights ###
tf.summary.histogram('W_input',parameters['w1'])
tf.summary.histogram('W_output',parameters['w2'])
#### End of histogram code code ####

parameters_for_later_use = parameters 

# function for forward pass
def two_layered_nn(data,hidden_size,parameters):
    """
    Function to build the two layered neural network to predict CIFAR10 data using tensor flow.
    Inputs: 
    data: input training data
    hidden_size: number of hidden neurons
    parameters: weights and biases    
   
    Output: 
    scores: scores computed after passing through the network
    """
  
  
    # Forward pass steps
    with tf.name_scope('input_layer'):
        input_relu = tf.add(tf.matmul(data,parameters['w1']),parameters['b1'], name = 'input_sum')
        output_relu = tf.nn.relu(input_relu,name = 'Relu_out')
        
    with tf.name_scope('output_layer'):
        scores = tf.add(tf.matmul(output_relu,parameters['w2']),parameters['b2'],name = 'output_sum')
        return(scores)
        
# function to train the neural network
def train(x,y,learning_rate,batch_size,hidden_size,train_,tr_target):
    """
    Function to train the two layered neural network to predict CIFAR10 data using tensor flow.
    Inputs: 
    x,y: tensor flow place holder for input training data
    train_,tr_target:input training data
    learning_rate: learning rate for optimisation
    batch_size= number of data samples processed for one pass in the network

    hidden_size: number of hidden neurons    
    """    
    scores = two_layered_nn(x,hidden_size,parameters)
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = scores, labels = y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        tf.summary.scalar("cost",cost)
    
    with tf.name_scope("accuracy"):
        correct = tf.equal(tf.argmax(scores,1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(correct,'float'))
        tf.summary.scalar("accuracy",acc)
    
    # Start a session to run the graph
    with tf.Session() as sess:
        log_writer = tf.summary.FileWriter("logs/",sess.graph) # defining the writer for tensorboard
        merged = tf.summary.merge_all() # initialize all summaries
        sess.run(tf.initialize_all_variables()) # initialize all variables
        
        for it in range(20000): 
            loss_history = []
            # generate random batches
            idx = np.random.choice(40000, batch_size, replace=True)
            ex = train_[idx]
            ey = tr_target[idx]  
            # run a session on optimizer and cost
            _,c = sess.run([optimizer,cost],feed_dict = {x:ex,y:ey}) 
            
            # run a session on 'merged' to get summaries
            if it%10 == 0:
                ### Type your code here ###
                s = sess.run(merged,feed_dict = {x:ex,y:ey})
                log_writer.add_summary(s,it)
                a = sess.run(acc, feed_dict = {x:ex,y:ey})
                
                #### End of your code ####
                log_writer.flush()
            loss_history.append(c)
            if it%50 == 0:
                print('iteration ',it, 'completed out of 20000, loss: ',c,' Training accuracy:', a)
        log_writer.close()
# Call the train function to buil and execute the graph
start_time = datetime.datetime.now()
train(x,y,learning_rate,batch_size,hidden_size,train_,tr_target)
print('Total execution time: '+ str(datetime.datetime.now() - start_time))