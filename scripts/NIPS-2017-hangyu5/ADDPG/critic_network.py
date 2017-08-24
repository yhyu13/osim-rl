import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from helper import *


LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
TAU = 0.001
L2 = 0.01

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,action_dim,scope):
        self.time_step = 0
        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net = self.create_q_network(state_dim,action_dim,scope)

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input,\
        self.target_q_value_output,\
        self.target_update = self.create_target_q_network(state_dim,action_dim,self.net,scope)

	if scope != 'global/critic':
            self.y_input = tf.placeholder("float",[None,1])
            weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
            self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
	    global_vars_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/critic')
	    local_vars_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
	    self.parameters_gradients,_ = zip(*self.optimizer.compute_gradients(self.cost,local_vars_critic))
	    self.parameters_graidents,_ = tf.clip_by_global_norm(self.parameters_gradients,5.0)
	    self.optimizer = self.optimizer.apply_gradients(zip(self.parameters_gradients,global_vars_critic))
            self.action_gradients = tf.gradients(self.q_value_output,self.action_input)
	    
	sess.run(tf.global_variables_initializer())

        #self.update_target()

    def create_q_network(self,state_dim,action_dim,scope):
	with tf.variable_scope(scope):

            state_input = tf.placeholder("float",[None,state_dim])
            action_input = tf.placeholder("float",[None,action_dim])
	    state_input2 = tf.reshape(state_input,shape=[-1,state_dim,1])
	    action_input2 = tf.reshape(action_input,shape=[-1,action_dim,1])

	    conv1 = tf.nn.elu(tf.nn.conv1d(state_input2,tf.truncated_normal([3,1,1],stddev=0.156),1,padding='VALID'))
	    conv2 = tf.nn.elu(tf.nn.conv1d(state_input2,tf.truncated_normal([5,1,1],stddev=0.156),1,padding='VALID'))
	    conv3 = tf.nn.elu(tf.nn.conv1d(state_input2,tf.truncated_normal([1,1,1],stddev=0.156),1,padding='VALID'))
	    conv4 = tf.nn.elu(tf.nn.conv1d(action_input2,tf.truncated_normal([3,1,1],stddev=0.236),1,padding='VALID'))
	    conv5 = tf.nn.elu(tf.nn.conv1d(action_input2,tf.truncated_normal([5,1,1],stddev=0.236),1,padding='VALID'))
	    conv6 = tf.nn.elu(tf.nn.conv1d(action_input2,tf.truncated_normal([1,1,1],stddev=0.236),1,padding='VALID'))
	    layer1 = slim.fully_connected(slim.flatten(tf.concat([conv1,conv2,conv3],axis=1)),300,activation_fn=tf.nn.elu,weights_initializer=tf.truncated_normal_initializer(stddev=0.05))
	    layer2 = slim.fully_connected(slim.flatten(tf.concat([conv4,conv5,conv6],axis=1)),200,activation_fn=tf.nn.elu,weights_initializer=tf.truncated_normal_initializer(stddev=0.05))
            q_value_output = slim.fully_connected(slim.flatten(tf.concat([layer1,layer2],axis=1)),1,activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.05))
	    net = [v for v in tf.trainable_variables() if scope in v.name]

            return state_input,action_input,q_value_output,net

    def create_target_q_network(self,state_dim,action_dim,net,scope):
        state_input,action_input,q_value_output,target_net = self.create_q_network(state_dim,action_dim,scope+'/target') 
        target_update = []
        for i in range(len(target_net)):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(target_net[i].assign(tf.add(tf.multiply(TAU,net[i]),tf.multiply((1-TAU),target_net[i]))))
        return state_input,action_input,q_value_output,target_update

    def update_target(self,sess):
        sess.run(self.target_update)

    def train(self,sess,y_batch,state_batch,action_batch):
        self.time_step += 1
        sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch
            })

    def gradients(self,sess,state_batch,action_batch):
        return sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch
            })[0]

    def target_q(self,sess,state_batch,action_batch):
        return sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def q_value(self,sess,state_batch,action_batch):
        return sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch})

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''