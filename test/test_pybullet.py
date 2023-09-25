"""
    [step 2] Test mecharm.urdf file and pybullet.
    Note: you should install pybullet correctly.
"""

import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0,0,-9.8)
p.setTimeStep(1./60)
p.setRealTimeSimulation(0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

startPos = [0,0,0]
planeId = p.loadURDF("plane.urdf")
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF('./myenv/envs/mecharm/mecharm_pi.urdf',startPos, startOrientation,useFixedBase = True)
numJoints = p.getNumJoints(boxId)

while(1):
    p.stepSimulation()
    time.sleep(1./240.)

pass