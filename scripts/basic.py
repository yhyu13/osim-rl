from osim.env.run import RunEnv
import numpy as np
from ou_noise import *
env = RunEnv(visualize=True)

def a1():
    a = np.ones(18)*0.05
    '''
    a[17:]=0.9
    a[0]=0.9
    a[3]=0.9
    a[4]=0.9
    a[8]=0.9
    a[11]=0.9
    a[12]=0.9
    a[13]=0.9
    a[10]=0.9
    '''
    a[8]=0.9
    a[9]=0.9
    a[12]=0.9
    a[13]=0.9
    a[17]=0.9
    a[2]=0.9
    a[3]=0.9
    #[4]=0.1
    a[1]=0.9
    a[0]=0.9
    return a

def normalize(s):
    s = np.asarray(s)
    s = (s-np.mean(s)) / np.std(s)
    return s
noise = OUNoise(18)
while True:
    observation = env.reset(difficulty = 2)
    a = a1()
    ob1, r, done, info =  env.step(a)
    i = 0
    while done == False:
        print(i)
        i += 1
        if i > 50:
            a += noise.noise()
        ob2, r, done, info = env.step(a)
        #print(normalize(np.asarray(ob2)-np.asarray(ob1)))
        ob1 = ob2
    env.reset()
    noise.reset()
    
