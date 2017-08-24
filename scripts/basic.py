from osim.env.run import RunEnv
import numpy as np

env = RunEnv(visualize=True)

observation = env.reset(difficulty = 0)
for i in range(2000000):
    a = np.ones(18)*0.05
    a[17:]=0.9
    a[0]=0.9
    a[3]=0.9
    a[4]=0.9
    a[8]=0.9
    a[11]=0.9
    a[12]=0.9
    a[13]=0.9
    a[10]=0.9
    print(i)
    observation, reward, done, info = env.step(a)
    if done:
        env.reset()
#        break
    

