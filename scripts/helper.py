import numpy as np
import random
import tensorflow as tf
import os
import csv
import itertools
import scipy.signal
import tensorflow.contrib.slim as slim

# Helper Function------------------------------------------------------------------------------------------------------------
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(s):
    #s = frame[10:-10,30:-30]
    s = (s-np.mean(s)) / np.std(s)
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

#This is a simple function to reshape our game frames.
def processState(state1):
    return np.reshape(state1,[21168])
    
#These functions allows us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars/2].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")
        
#Record performance metrics and episode logs for the Control Center.
def saveToCenter(i,rList,jList,bufferArray,summaryLength,h_size,sess,mainQN,time_per_step):
    with open('./Center/log.csv', 'a') as myfile:
        state_display = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        imagesS = []
        for idx,z in enumerate(np.vstack(bufferArray[:,0])):
            img,state_display = sess.run([mainQN.salience,mainQN.rnn_state],\
                feed_dict={mainQN.scalarInput:np.reshape(bufferArray[idx,0],[1,21168])/255.0,\
                mainQN.trainLength:1,mainQN.state_in:state_display,mainQN.batch_size:1})
            imagesS.append(img)
        imagesS = (imagesS - np.min(imagesS))/(np.max(imagesS) - np.min(imagesS))
        imagesS = np.vstack(imagesS)
        imagesS = np.resize(imagesS,[len(imagesS),84,84,3])
        luminance = np.max(imagesS,3)
        imagesS = np.multiply(np.ones([len(imagesS),84,84,3]),np.reshape(luminance,[len(imagesS),84,84,1]))
        make_gif(np.ones([len(imagesS),84,84,3]),'./Center/frames/sal'+str(i)+'.gif',duration=len(imagesS)*time_per_step,true_image=False,salience=True,salIMGS=luminance)

        images = zip(bufferArray[:,0])
        images.append(bufferArray[-1,3])
        images = np.vstack(images)
        images = np.resize(images,[len(images),84,84,3])
        make_gif(images,'./Center/frames/image'+str(i)+'.gif',duration=len(images)*time_per_step,true_image=True,salience=False)

        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([i,np.mean(jList[-100:]),np.mean(rList[-summaryLength:]),'./frames/image'+str(i)+'.gif','./frames/log'+str(i)+'.csv','./frames/sal'+str(i)+'.gif'])
        myfile.close()
    with open('./Center/frames/log'+str(i)+'.csv','w') as myfile:
        state_train = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["ACTION","REWARD","A0","A1",'A2','A3','V'])
        a, v = sess.run([mainQN.Advantage,mainQN.Value],\
            feed_dict={mainQN.scalarInput:np.vstack(bufferArray[:,0])/255.0,mainQN.trainLength:len(bufferArray),mainQN.state_in:state_train,mainQN.batch_size:1})
        wr.writerows(zip(bufferArray[:,1],bufferArray[:,2],a[:,0],a[:,1],a[:,2],a[:,3],v[:,0]))
    



