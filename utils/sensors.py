import numpy as np
import torch as th
import gymnasium as gym
import pytorch_kinematics as pk


class BaseSensor():
    """
       The base class for Sensor: y = g(x,w)
    """
    def __init__(self, input_dim:int) -> None:
        self.input_dim = input_dim
        self.output_dim = self.__call__(x=th.zeros((1,input_dim))).shape[2]
        pass
    
    def __call__(self, x:th.Tensor) -> th.Tensor:
        """ Note that x.shape = [n, input_dim] """
        return self.sensor_sim(x, sim_num=1) 

    def sensor_sim(self,  x:th.Tensor,  sim_num:int=int(10e4)) -> th.Tensor:
        """ Return ouput (th.Tensor): The ouput of (noisy) sensor.
            Note that x.shape = [n, input_dim], y.shape = [n, sim_num, ouput_dim]
            :param state (th.Tensor): The true state.
            :param sim_num (int): repeat sensoring sim_num times.
        """
        output_dim = 1
        return th.zeros((len(x),sim_num,output_dim))

class LinearGaussianSensor(BaseSensor):
    '''
        The sensor model:
        y = Cx + Gw, w ~ N(0,1)
    ''' 
    def __init__(self, C:np.ndarray, G:np.ndarray):
        self.G = th.tensor(G, dtype=th.float32)
        self.C = th.tensor(C, dtype=th.float32)
        self.n_w = self.G.shape[0]
        super().__init__(input_dim=self.C.shape[1]) 

    def sensor_sim(self, x:th.Tensor,  sim_num:int=int(10e4))->th.Tensor:
        '''
            As we know the model of the sensor and the distribution of noise
            w, then we can obtian y = Cx + G*w, w ~ N(0,1) by simulation

            :param sim_num : how many times of the simulation.
        '''
        x = x.unsqueeze(dim=1).transpose(dim0=1,dim1=2)
        w = th.randn(size=(sim_num,len(x),self.n_w,1),dtype=th.float32)
        return (self.C.to(x.device) @ x + self.G.to(x.device) @ w.to(x.device)).transpose(dim0=0,dim1=1)


class ForwardKinematicsGaussianSensor(BaseSensor):
    '''
        The sensor model:
        y = FK(x) + G*w, w ~ N(0,1)
    '''
    def __init__(self, IsRot=False):
        self.IsRot = IsRot
        if IsRot:
            self.G = th.tensor([0.1,0.1,0.1,0.,0.,0.,0.], dtype=th.float32)
        else:
            self.G = th.ones((3,), dtype=th.float32)*0.05   
        self.robot = pk.build_serial_chain_from_urdf(
    open('./myenv/envs/mecharm/mecharm_pi.urdf').read(),end_link_name='link6')
        self.n_w = self.G.shape[0]
        input_dim = len(self.robot.get_joint_parameter_names())
        super().__init__(input_dim=input_dim)

    def sensor_sim(self, x:th.Tensor,  sim_num:int=int(10e4))->th.Tensor:
        '''
            As we know the model of the sensor and the distribution of noise
            w, then we can obtian y = FK(x) + G*w, w ~ N(0,1) by simulation
            y.shape = (len(x),sim_num,output_dim)

            :param sim_num : how many times of the simulation.
        '''
        self.robot.to(dtype=x.dtype,device=x.device)
        fk = self.robot.forward_kinematics(x)

        m = fk.get_matrix()
        pos = m[:, :3, 3]
        w = th.randn(size=(sim_num,len(x),self.n_w),dtype=th.float32)
        if self.IsRot:
            rot = pk.matrix_to_quaternion(m[:, :3, :3])
            return (th.hstack([pos,rot]) + self.G.to(x.device) * w.to(x.device)).transpose(dim0=0,dim1=1)
        else:
            return (pos + self.G.to(x.device) * w.to(x.device)).transpose(dim0=0,dim1=1)
