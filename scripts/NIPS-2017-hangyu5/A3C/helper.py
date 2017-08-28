import numpy as np
import tensorflow as tf
import scipy.signal as ss
from multiprocessing import Process, Pipe

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

# Normalize state 
def process_frame(s):
    s = np.asarray(s)
    s = (s-np.mean(s)) / np.std(s)
    return s

# process state (the last 3 entires are obstacle info which should not be processed)
def process_state(s,s1):
    s = np.asarray(s)
    s1 = np.asarray(s1)
    s = np.hstack((s1[:-3]-s[:-3],s[-3:]))
    return s
    
def engineered_action(seed):
    a = np.ones(18)*0.05
    if seed < .5:
        a[17:]=0.9
        a[0]=0.9
        a[3]=0.9
        a[4]=0.9
        a[8]=0.9
        a[11]=0.9
        a[12]=0.9
        a[13]=0.9
        a[10]=0.9
    else:
        a[8]=0.9
        a[9]=0.9
        a[12]=0.9
        a[13]=0.9
        a[17]=0.9
        a[2]=0.9
        a[3]=0.9
        a[4]=0.9
        a[1]=0.9 
    return a

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def discount(x, gamma):
    return ss.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

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
        
# [Hacked] the memory might always be leaking, here's a solution #58
# https://github.com/stanfordnmbl/osim-rl/issues/58 
# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def floatify(np):
    return [float(np[i]) for i in range(len(np))]
    
def standalone_headless_isolated(conn,vis):
    e = RunEnv(visualize=vis)
    
    while True:
        try:
            msg = conn.recv()

            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':
                o = e.reset(difficulty=2)
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
	except EOFError:  
            return None

    def __del__(self):
        self.pc.send(('exit',))
        #print('(ei)waiting for join...')
        self.p.join()
    



