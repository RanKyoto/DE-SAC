"""
    [step 5] Test mecharm_pi.urdf file and pytorch_kinematics.
    Note: you should install pytorch_kinematics correctly.
          pip install pytorch-kinematics
    URL: https://github.com/UM-ARM-Lab/pytorch_kinematics
    Bibtex:
    @software{Zhong_PyTorch_Kinematics_2023,
        author = {Zhong, Sheng and Power, Thomas and Gupta, Ashwin},
        doi = {10.5281/zenodo.7700588},
        month = {3},
        title = {{PyTorch Kinematics}},
        version = {v0.5.4},
        year = {2023}
        }
"""

import pytorch_kinematics as pk

import torch as th

robot = pk.build_serial_chain_from_urdf(
    open('./myenv/envs/mecharm/mecharm_pi.urdf').read(),end_link_name='link6')
# show links and joints
print(robot)
print(robot.get_joint_parameter_names())

ee = robot.forward_kinematics(th.zeros((6,)))
print(ee)
print(ee.get_matrix())

# test pytorch
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
robot.to(dtype=th.float32, device= device)
# generate N samples of the angles of the joints of the robot
N = 10
joints_batch = th.rand(N, len(robot.get_joint_parameter_names()), 
            dtype=th.float32, device=device)
# obtain the states of the end-effector
ee_batch = robot.forward_kinematics(joints_batch)
a = ee_batch[0]
T = a.get_matrix()
pos = T[:,:3,3]
rot = pk.matrix_to_quaternion(T[:,:3,:3])
print(pos,rot)


