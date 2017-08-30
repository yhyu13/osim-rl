# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Kaizhao Liang, Hang Yu
# Date: 08.21.2017
# -----------------------------------
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from replay_buffer import ReplayBuffer
from helper import *
from time import gmtime, strftime, sleep

import opensim as osim
from osim.http.client import Client
from osim.env import *

import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


import multiprocessing
from multiprocessing import Process, Pipe

# [Hacked] the memory might always be leaking, here's a solution #58
# https://github.com/stanfordnmbl/osim-rl/issues/58 
# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def standalone_headless_isolated(conn,vis):
    with HiddenPrints():
       e = RunEnv(visualize=vis)
    
    while True:
        try:
            msg = conn.recv()

            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':
                o = e.reset(difficulty=1)
                conn.send(o)
            elif msg[0] == 'step':
                ordi = e.step(msg[1])
                conn.send(ordi)
            else:
                conn.close()
                del e
                return
        except:
            conn.close()
            del e
            raise

# class that manages the interprocess communication and expose itself as a RunEnv.
class ei: # Environment Instance
    def __init__(self,vis):
        self.pc, self.cc = Pipe()
        self.p = Process(
            target = standalone_headless_isolated,
            args=(self.cc,vis,)
        )
        self.p.daemon = True
        self.p.start()

    def reset(self):
        self.pc.send(('reset',))
        return self.pc.recv()

    def step(self,actions):
        self.pc.send(('step',actions,))
        try:
            return self.pc.recv()
        except :  
            print('Error in recv()')
            pass

    def __del__(self):
        self.pc.send(('exit',))
        #print('(ei)waiting for join...')
        self.p.join()
	del self.pc
	del self.cc
	del self.p

###############################################
# DDPG Worker
###############################################
pause_perceive = False

class Worker:
    """docstring for DDPG"""
    def __init__(self,sess,number,model_path,global_episodes,explore,training,vis,ReplayBuffer,batch_size,gamma,replay_buffer_size):
        self.name = 'worker_' + str(number) # name for uploading results
        self.number = number
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = 41+3+14 # 41 observations plus 17 induced velocity
        self.action_dim = 18
        self.model_path= model_path
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.sess = sess
        self.explore = explore
        self.noise_decay = 1.
        self.training = training
        self.vis = vis # == True only during testing
        self.total_steps = 0 # for ReplayBuffer to count
        self.batch_size = batch_size
        self.gamma = gamma
	self.replay_buffer_size = replay_buffer_size

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim,self.name+'/actor')
        self.actor_network.update_target(self.sess)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim,self.name+'/critic')
        self.critic_network.update_target(self.sess)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        self.update_local_ops_actor = update_graph('global/actor',self.name+'/actor')
        self.update_local_ops_critic = update_graph('global/critic',self.name+'/critic')
        self.update_local_ops_actor_target = update_graph('global/actor/target',self.name+'/actor/target')
        self.update_local_ops_critic_target = update_graph('global/critic/target',self.name+'/critic/target')
        self.update_global_actor_target = update_target_network(self.name+'/actor/target','global/actor/target',1e-4)
        self.update_global_critic_target = update_target_network(self.name+'/critic/target','global/critic/target',1e-4)

    def start(self):
        self.env = ei(vis=self.vis)#RunEnv(visualize=True)
            
    def restart(self): # restart env every ? eps to coutner memory leak
        if self.env != None:
            del self.env
	    sleep(0.1)
        self.env = ei(vis=self.vis)

    def train(self):
        # print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        tree_idx, minibatch, ISWeights = self.replay_buffer.sample(self.batch_size)
        #print(ISWeights)
	#self.batch_size = len(minibatch)
	print(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
	    # reward clipping:  scale and clip the values of the rewards to the range -1,+1
	    # reward_batch = (reward_batch - np.mean(reward_batch)) / np.max(abs(reward_batch))

        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[self.batch_size,self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(self.sess,next_state_batch)
        q_value_batch = self.critic_network.target_q(self.sess,next_state_batch,next_action_batch)
        #y_batch = []
        done_mask = [0 if done else 1 for done in done_batch]
        '''
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + self.gamma * q_value_batch[i])'''
        y_batch = reward_batch + self.gamma * q_value_batch * done_mask
        y_batch = np.resize(y_batch,[self.batch_size,1])
        # Update critic by minimizing the loss L
        _,abs_errors,loss,a,b,norm = self.critic_network.train(self.sess,y_batch,state_batch,action_batch,ISWeights)
        print(a)
        print(b)
        print(loss)
        print(norm)
        self.replay_buffer.batch_update(tree_idx, abs_errors)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(self.sess,state_batch)
        q_gradient_batch = self.critic_network.gradients(self.sess,state_batch,action_batch_for_gradients)

        _,norm = self.actor_network.train(self.sess,q_gradient_batch,state_batch)
        print(norm)
        # Update the target networks
        #self.actor_network.update_target(self.sess)
        #self.critic_network.update_target(self.sess)
        self.sess.run(self.update_global_actor_target)
        self.sess.run(self.update_global_critic_target)
        self.sess.run(self.update_local_ops_actor)
        self.sess.run(self.update_local_ops_critic)
        self.sess.run(self.update_local_ops_actor_target)
        self.sess.run(self.update_local_ops_critic_target)

    def save_model(self, saver, episode):
        saver.save(self.sess, self.model_path + "/model-" + str(episode) + ".ckpt")

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise which gradually vanishes
        action = self.actor_network.action(self.sess,state)
        return action+self.exploration_noise.noise()*self.noise_decay

    def action(self,state):
        action = self.actor_network.action(self.sess,state)
        return action

    def perceive(self,state,action,reward,next_state,done,action_avg,step,ea):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        transition = [state, action, reward, next_state, done]
        self.replay_buffer.store(transition)
        self.total_steps += 1

        # if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset(None) # reset as noise mu as (1- average_activation_each_muscle)

    def work(self,coord,saver):
        if self.training:
            episode_count = self.sess.run(self.global_episodes)
        else:
            episode_count = 0
        wining_episode_count = 0
        
	global pause_perceive

        print ("Starting worker_" + str(self.number))
        
        if self.name == 'worker_0':
            with open('result.txt','w') as f:
                f.write(strftime("Starting time: %a, %d %b %Y %H:%M:%S\n", gmtime()))
        
        self.start() # change Aug24 start the env

        with self.sess.as_default(), self.sess.graph.as_default():
            #not_start_training_yet = True
            while not coord.should_stop():
            
                if episode_count % 5 == 0 and episode_count>1: # change Aug24 restart RunEnv every 50 eps
                    self.restart()
                    
                returns = []
                episode_reward = 0
                self.noise_decay = np.cos(self.explore / 100 * 2* np.pi)
                #print(self.noise_decay)
                self.explore -= 1
                
                self.sess.run(self.update_local_ops_actor)
                self.sess.run(self.update_local_ops_critic)
                self.sess.run(self.update_local_ops_actor_target)
                self.sess.run(self.update_local_ops_critic_target)
                        
                state = self.env.reset()
                #print(observation)
                seed= 0.1#np.random.rand()
                ea=engineered_action(seed)
                
                
                s,s1,s2 = [],[],[]
                ob = self.env.step(ea)[0]
                s = ob
                #print(s[1:3],s[26:28]) # plvis position appear twice? test says no. https://github.com/stanfordnmbl/osim-rl search "41 observation"
                ob = self.env.step(ea)[0]
                s1 = ob
                #print(s1[1:3],s1[26:28])
                s = process_state(s,s1)
                    
                if self.name == 'worker_0':
                    print("episode:{}".format(str(episode_count)+' '+self.name))
                # Train
                done = False
                chese = 100#int(np.random.rand()*50) # change Aug 25 >50 == turn off engineered action
                action_avg = np.zeros(18)
                for step in xrange(1000):
                    if chese < 70:
                        action=engineered_action(seed)
                        #action_avg += action
                        action = np.clip(action+self.exploration_noise.noise()*0.0,0.0,1.0)
                        chese += 1
                    elif self.explore>0 and self.training:
                        action = np.clip(self.noise_action(s),-1e-2,1.0-1e-2) # change Aug20
                        #action_avg += action
                    else:
                        action = self.action(s)
                        #action_avg += action
                    try:
                        s2,reward,done,_ = self.env.step(action)
                    except:
                        print('Env error. Shutdown {}'.format(self.name))
                        if self.env != None:
			    del self.env
                        return 0
                    #print('state={}, action={}, reward={}, next_state={}, done={}'.format(state, action, reward, next_state, done))
                    s1 = process_state(s1,s2)
                    #if chese >=50: # change Aug24 do not include engineered action in the buffer
                        #self.perceive(s,action,reward,s1,done)
		    sleep(0.001) # THREAD_DELAY
                    if step % 3 == 0 and not pause_perceive:
                        self.perceive(s,normalize(action),reward*20,s1,done,action_avg,step,ea)
                        
                    if self.name == "worker_1" and self.total_steps > 100 and self.training:
			pause_perceive=True
			#print(self.name+'is training')
                        self.train()
                        self.train()
			pause_perceive=False

                    s = s1
                    s1 = s2
                    episode_reward += reward
                    if done:
                    	break

                if self.name == 'worker_0' and episode_count % 5 == 0:
                    with open('result.txt','a') as f:
                        f.write("Episode "+str(episode_count)+" reward (training): %.2f\n" % episode_reward)


                # Testing:
                if self.name == 'worker_2' and episode_count % 10 == 0 and episode_count > 1: # change Aug19
                    if episode_count % 50 == 0 and not self.vis:
                        self.save_model(saver, episode_count)
               	    total_return = 0
                    for i in xrange(3):
                        state = self.env.reset()
               	        a=engineered_action(seed)
                        ob = self.env.step(a)[0]
                        s = ob
                        ob = self.env.step(a)[0]
                        s1 = ob
                        s = process_state(s,s1)
                        for j in xrange(1000):
                            action = self.action(s) # direct action for test
                            s2,reward,done,_ = self.env.step(action)
                            s1 = process_state(s1,s2)
                            s = s1
			    s1 = s2
                            total_return += reward
                            if done:
                                break

                    ave_return = total_return/3
                    returns.append(ave_return)
                    with open('result.txt','a') as f:
                        f.write('episode: {} Evaluation(testing) Average Return: {}\n'.format(episode_count,ave_return))

                if self.name == 'worker_0' and self.training:
                    self.sess.run(self.increment)
                episode_count += 1

            # All done Stop trail
            # Confirm exit
            print('Done '+self.name)
            return








