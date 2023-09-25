"""
    [step 4] Test stable-baselines3.
    Note: you should install stable_baselines3 correctly
          e.g.: pip install stable-baselines3
"""

import sys
sys.path.append('./')
import gymnasium as gym
import myenv
import numpy as np
from stable_baselines3 import SAC,DDPG

env = gym.make("MechArm-v0",render_mode='human')
#env = gym.make("MechArm-v0",render_mode='rgb_array')

model =  SAC('MlpPolicy', env, verbose=1,
                     buffer_size=int(10e5),batch_size=256,learning_rate=0.001)
model.learn(100000,log_interval=1)