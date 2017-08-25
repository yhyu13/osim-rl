from ddpg import *
import opensim as osim
from osim.http.client import Client
from osim.env import *
import os

ENV_NAME = 'learning_to_run'
MODEL_PATH = './models'
EPISODES = 100000
TEST = 3
load_model = True # change Aug20

def process_frame(s):
    s = (s-np.mean(s)) / np.std(s)
    return s

def main():
    env = RunEnv(visualize=True)
    env.reset(difficulty = 0)
    agent = DDPG(env)

    if not os.path.exists(MODEL_PATH): # change Aug19
        os.makedirs(MODEL_PATH)

    returns = []
    rewards = []
    tf.reset_default_graph() # change Aug19

    EXPLORE = 2000 # change Aug19
    ep_restart = 0

    if load_model:
	agent.load_model(MODEL_PATH)
	EXPLORE -= 401*1
	ep_restart = 401

    for episode in xrange(ep_restart,EPISODES): # change Aug20
        state = env.reset(difficulty = 0)
        reward_episode = 0
        print "episode:",episode

	if np.random.rand() < 0.9: # change Aug20 stochastic apply noise
	    EXPLORE -= 1

        for step in xrange(env.spec.timestep_limit):
	    state = process_frame(state)
	    if EXPLORE>0:
                action = np.clip(agent.noise_action(state),0.0,1.0) # change Aug19
	    else:
		action = agent.action(state)
            next_state,reward,done,_ = env.step(action)
            #print('state={}, action={}, reward={}, next_state={}, done={}'.format(state, action, reward, next_state, done))
	    next_state = process_frame(next_state)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            reward_episode += reward
            if done:
		#print "Done before 1000 iterations! Episode reward:",reward_episode                
		break
		
	if episode % 5 == 0:
	    print "episode reward:",reward_episode
		

        # Testing:
        #if episode % 1 == 0:
        if episode % 50 == 0 and episode > 1: # change Aug19
            agent.save_model(MODEL_PATH, episode)

            total_return = 0
            ave_reward = 0
            for i in xrange(TEST):
                state = env.reset(difficulty=1)
                reward_per_step = 0
                for j in xrange(env.spec.timestep_limit):
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_return += reward
                    if done:
                        break
                    reward_per_step += (reward - reward_per_step)/(j+1)
                ave_reward += reward_per_step

            ave_return = total_return/TEST
            ave_reward = ave_reward/TEST
            returns.append(ave_return)
            rewards.append(ave_reward)

            print 'episode: ',episode,'Evaluation Average Return:',ave_return, '  Evaluation Average Reward: ', ave_reward

if __name__ == '__main__':
    main()
