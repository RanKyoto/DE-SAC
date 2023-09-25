"""
    [step 1] Test Linear-v0 envronment.
    Note: you should install gymnasium correctly
          e.g.: pip install gymnasium
"""

import sys
sys.path.append('./')
import gymnasium as gym
import myenv
import numpy as np

env = gym.make("Linear-v0",render_mode='human')

n=10
for ep in range(n):
    #env.reset(options={"x0":np.array([7.,7.])})
    env.reset()
    k = 0.03
    action = np.tanh(k * env.output)*20
    for i in range(300):
        state, reward, done, _,_ = env.step(action=action)
        action =  np.tanh(k * env.output)*20
