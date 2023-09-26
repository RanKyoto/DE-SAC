from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class MechArmEnv(gym.Env):
    """
    Description:
        Note: This is a discrete-time Linear noisy output feedback control system.

    States:
        Type: Box(6)
        Num     States            Min               Max
        0~5      x[i]           - 1   rad           1 rad

    Actions:
        Type: Box(6)
        Num     Action            Min               Max
        0~5      u[i]           - 1                 1

    System:
        x' = f(x,u)    # the next state of robotic arm
        y  = g(x,w)      # the position of end-effector
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        'render_fps':30
        }
    def __init__(self, render_mode: Optional[str] = None):
        if render_mode=='human':
            renderer = 'OpenGL'
        else:
            renderer = 'Tiny'

        self.sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.sim.create_plane(z_offset=-0.2)
        self.sim.create_table(length=0.6, width=0.34, height=0.2, x_offset=-0.2)

        self.robot = MechArm(sim=self.sim,control_type="joints")
        self.render_mode = render_mode

        # the neutral position of the end-effector [m]
        # array([1.68000004e-01, 3.85686529e-07, 2.43000000e-01])
        self.target_position = self.robot.get_ee_position() 

        self.dt = self.sim.dt  # sample period

        self.observation_space =spaces.Box(-2.0, 2.0, shape=(6,), dtype=np.float32)
        self.action_space = self.robot.action_space

        #render settings
        self.render_width= 720
        self.render_height = 480
        self.render_distance = 0.5
        self.render_yaw = 45
        self.render_pitch = -30
        self.render_roll = 0

        with self.sim.no_rendering():
            self.sim.place_visualizer(
                target_position= np.zeros((3,)),
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

        self.sim.create_sphere(
            body_name="noisy_outputs",
            radius=0.01,
            mass=0.0,
            ghost=True,
            position=self.robot.get_ee_position(),
            rgba_color=np.array([0.0, 0.3, 1.0, 0.6]),
        )

        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=self.robot.get_ee_position(),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

        self.sim.set_base_pose("target",position=self.robot.get_ee_position(),orientation=[0,-0.71,0,0.71])

        self.state = None  # state  x_k
        self.output = None # output y_k

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self,action)->float:
        ''' 
            return the true reward
            r(x) = - (10*|ee_pos|_2 + |u|_2)
        '''
        
        # typical quadratic reward function
        return -self.dt*(10*np.linalg.norm(self.output[0:3]-self.target_position) +  np.linalg.norm(action))
        

    def step(self, action:np.ndarray):
        '''
            Discrete-time control system, x'= x+ 0.05u;  
            Return: x, r(x,u), False, False, {}
        '''
        self.robot.set_action(action)
        self.sim.step()
        self.state = self._get_obs()
        self.output = self._get_output()
        noisy_outputs = self.output[0:3] + 0.05*np.random.randn(3)
        # An episode is terminated if the agent has reached the target
        terminated = False
        truncated = False
        info = {}
        reward = self.reward(action)
        self.sim.set_base_pose("noisy_outputs",position=noisy_outputs,orientation=[0,0,0,1])
        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = {"random":True, "x0":None}):
        super().reset(seed=seed)
        if options['random']: 
            self.robot.neutral_joint_values = self.observation_space.sample()*0.5
        else:
            if options['x0'] is None:
                self.robot.neutral_joint_values = np.zeros((6,))
            else:
                self.robot.neutral_joint_values = np.array(options['x0'])
                
        with self.sim.no_rendering():
            self.robot.reset()
        self.state = self._get_obs()
        self.output = self._get_output()

        return self.state, {}

    def _get_obs(self) -> np.ndarray:
        """ obtain the true state x -- the angles of joints """
        robot_obs = self.robot.get_obs(obs_type='joints').astype(np.float32)  # robot state
        return robot_obs

    def _get_output(self):
        '''
            obtain the output y -- the position of the end-effector
        '''
        ee_pos=self.robot.get_ee_position() 
        ee_rot=self.robot.get_ee_orientation()
        return np.hstack([ee_pos,ee_rot]) # return y

    def render(self):
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """

        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=np.zeros((3,)),
            distance=0.5,
            yaw=45,
            pitch=-30,
            roll=0,
        )

    def close(self) -> None:
        self.sim.close()


class MechArm(PyBulletRobot):
    """MechArm270 robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 6  # control (x, y z) if "ee", else, control the 6 joints
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="mecharm_pi",
            file_name="myenv/envs/mecharm/mecharm_pi.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5]),
            joint_forces=np.ones((6,))*87,
        )
        self.ee_link = 5 # index of the end-effector
        self.neutral_joint_values = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
       

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:6]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        
        self.control_joints(target_angles=target_arm_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:6]  
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 6 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 6 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(6)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self,obs_type = 'joints') -> np.ndarray:
        if obs_type == "joints":
            # angles of the joints
            observation = np.array([self.get_joint_angle(i) for i in range(6)])
        elif obs_type == "ee":
            # end-effector position and velocity
            ee_position = np.array(self.get_ee_position())
            ee_velocity = np.array(self.get_ee_velocity())

            observation = np.concatenate((ee_position, ee_velocity))
        else:
             assert False, "obs_type must be joints or ee"
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)


    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_orientation(self) -> np.ndarray:
        """Returns the orientation of the end-effector as quaternion (x, y, z, w)."""
        return self.sim.get_link_orientation(self.body_name,self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

