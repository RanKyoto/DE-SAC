import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from gymnasium.error import DependencyNotInstalled
from typing import Optional

class LinearEnv(gym.Env):
    """
    Description:
        Note: This is a discrete-time Linear noisy output feedback control system.

    States:
        Type: Box(2)
        Num     States            Min               Max
        0       x1                - 10               10
        1       x2                - 10               10
    Note: Output y ∈ (-∞，∞) 

    Actions:
        Type: Box(1)
        Num     Action            Min               Max
        0       u                 - 10               10

    System:
        x' = Ax + Bu + Fv ~ (0, 1) 
        y  = Cx + Gw, w ~ N(0,1)
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        }
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode

        self.dt = 0.01  # sample period

        # Initial system parameters
        A =np.array( [[0.99040   ,  0.03921 ],  
                      [0.01961   ,  0.97080 ]])
        B = np.array([[0.02069],
                      [0.03961]])
        C = np.array([[-2., 1.]])

        G = np.array( [[1.0]])

        F = np.array( [[0.2],  
                       [0.1]])

        self.Q = np.array( [[1  ,  0  ],  
                            [0  ,  1  ]])
        self.R = 0.1

        # Check the dimensions 
        self.A = A.reshape(2,2)     
        self.B = B.reshape(2,1)
        self.C = C.reshape(1,2)
        self.G = G.reshape(1,1)
        self.F = F.reshape(2,1)
        self.V = self.F @ self.F.T
        self.W = self.G @ self.G.T

        # Boundary settings
        self.u_mag = 20.0 # max input magnitude
        self.x_max = 15.0 # max state magnitude

        self.action_space = spaces.Box(
            low= -self.u_mag,
            high= self.u_mag,
            shape=(1,), 
            dtype=np.float32
            )
        self.observation_space = spaces.Box(
            low= -self.x_max * np.ones(2,dtype=np.float32), 
            high= self.x_max * np.ones(2,dtype=np.float32), 
            dtype=np.float32
            )

        self.screen_width = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None  # state  x_k
        self.output = None # output y_k

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self,u)->float:
        ''' 
            return the true reward
            r(x,u) = - (x'Qx + u'Ru)  (True reward)  
        '''
        x = np.array(self.state).reshape(2,1)
        # typical quadratic reward function
        return -float(x.T @ self.Q @ x + self.R * (u**2))*self.dt 
        

    @property       
    def optimal_action(self)->np.ndarray:
        """ 
            optimal state feedback control gain. (NOTE: not the output feedback)
        """
        import control
        x = self.state.reshape(2,1)
        K,_,_ = control.dlqr(self.A,self.B,self.Q,self.R)
        K = K.reshape(1,2)
        return (-K @ x).flatten()

    def step(self, action:np.ndarray):
        '''
            Discrete-time control system, x'= Ax + Bu + v;  
            Return: x, r(x,u), IsOutOfRange, False, {}
        '''
        u = np.clip(action, -self.u_mag, self.u_mag)

        reward = self.reward(u)

        self.output = self._get_output() # y = Cx+Gw

        # x' = Ax + Bu + Fv
        # V = GG'
        v = np.random.multivariate_normal(np.zeros(self.state.shape),self.V).reshape(2,1)
        x_prime = self.A @ self.state.reshape(2,1) +  self.B * u + v
        self.state = np.array(x_prime.flatten(), dtype=np.float32) 
        

        # If state is out of space, terminated = True
        terminated = not self.observation_space.contains(self.state)
        #terminated = False

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False , {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        ''' reset with x0 ~ d0 or some given x0'''
        if options is None:
            random_x1 = np.random.uniform(-8,8)
            random_x2 = np.random.uniform(-8,8)
        else:
            random_x1 = options['x0'][0]
            random_x2 = options['x0'][1]

        self.state = np.array([random_x1,random_x2],dtype=np.float32)
        self.output = self._get_output()

        return self.state, {}

    def _get_output(self):
        '''
            y = Cx + Gw, w ~ N(0,1) --> y ~ N(Cx,GG'), W = GG'
        '''
        Cx = self.C @ self.state
        return np.random.multivariate_normal(Cx,self.W) # return y

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
            
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:     
            raise DependencyNotInstalled(
            "pygame is not installed, run `pip install gym[classic_control]`"
        )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                        (self.screen_width, self.screen_width)
                    )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_width))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_width))
        self.surf.fill((255, 255, 255))
        
        scale = self.screen_width / (self.x_max * 2)
        offset = self.screen_width // 2

                
        gfxdraw.hline(self.surf, 0, self.screen_width, offset, (0, 0, 0))
        gfxdraw.vline(self.surf, offset, 0, self.screen_width, (0, 0, 0))

        x1,x2 = self.state
        x1 = int(x1 * scale + offset)
        x2 = int(x2 * scale + offset)
        gfxdraw.filled_circle(self.surf,x1,x2,5,(20,20,255))

        y = int(self.output * scale / 2)
        gfxdraw.vline(self.surf,x1,x2,x2+y, (255,125,125))
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
 

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
