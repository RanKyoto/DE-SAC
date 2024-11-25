"""
    [step 3] Test MechArm-v0 envronment.
    Note: you should install pandagym correctly
          e.g.: pip install pandagym
"""

import sys
sys.path.append('./')
import gymnasium as gym
import myenv
import numpy as np

env = gym.make("MechArm-v0",render_mode='human')
env.reset(options={"random":False,"x0":None})
print("Joints:",env.unwrapped.state,
      "ee-Position:",env.unwrapped.output[:3],
      'ee-Orientation:',env.unwrapped.output[3:])
n=100
for ep in range(n):
    state,_ = env.reset(options={"random":True})
    for i in range(3000):
        action = -0.03*env.unwrapped.robot.inverse_kinematics(5,env.unwrapped.output[:3],env.unwrapped.output[3:])
        state, reward, done, _,_ = env.step(action=action)
 
