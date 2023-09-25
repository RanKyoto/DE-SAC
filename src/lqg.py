import sys
sys.path.append('./')

import gymnasium
import myenv
import control
import numpy as np
from utils import process_bar

class LQG():
    def __init__(self) -> None:
        env = gymnasium.make("Linear-v0",render_mode='rgb_array')
        A = env.A
        B = env.B
        F = env.F
        C = env.C
        G = env.G
        #U = env.u_mag
        Q = env.Q
        R = env.R

        self.env = env
        self.dt = env.dt
        self.A = A
        self.B = B
        self.C = C
        # feedback gain
        self.K = control.dlqr(A,B,Q,R)[0]
        # kalman gain
        self.L = control.dlqe(A,np.eye(2),C, F@F.T, G@G.T)[0]
        # equivlent to: L = control.dlqe(A,F,C, 1, G@G.T)[0]

    def simulation(self, n_step = 500, isPlot:bool = False)->float:
        x,_ = self.env.reset()
        #x_hat = np.zeros((2,1))
        x_hat = x.reshape(2,1)
        ep_reward = 0.0
        if isPlot:
            x1_list,x2_list,y_list,x1_hat_list,x2_hat_list, u_list = [],[],[],[],[],[]
        for _ in range(n_step):
            y = self.env.output
            u = -self.K @ x_hat
            x_hat = self.A @ x_hat.reshape(2,1) + self.B * u + self.L @ (y - self.C @ x_hat.reshape(2,1))
            
            x,reward,_,_,_ = self.env.step(u)
            ep_reward += reward
            if isPlot:
                x1_list.append(x[0])
                x2_list.append(x[1])
                y_list.append(y[0])
                x1_hat_list.append(x_hat[0])
                x2_hat_list.append(x_hat[1])
                u_list.append(u[0])

        if isPlot:
            t_list = np.linspace(0,n_step*self.dt,n_step)
            import matplotlib.pyplot as plt
            fig,axes = plt.subplots(2,2,figsize = (16,9))
            axes[0,0].plot(t_list,x1_list)
            axes[0,0].plot(t_list,x1_hat_list)

            axes[0,1].plot(t_list,x2_list)
            axes[0,1].plot(t_list,x2_hat_list)

            axes[1,0].plot(t_list,y_list)
            axes[1,1].plot(t_list,u_list)
            fig.tight_layout()
            plt.show()

        return ep_reward

    def mean_reward(self,n_ep:int = 1000):
        reward_list = []
        for ep in range(n_ep):
            mean_reward = self.simulation(n_step=300)
            reward_list.append(mean_reward)
            process_bar((ep+1)/n_ep,end_str="|reward={}".format(np.mean(reward_list)))
        return np.mean(reward_list)

# testing code
if __name__ == '__main__':   
    lqg_ctr = LQG()
    mean_reward = lqg_ctr.simulation(isPlot=True)
    lqg_ctr.mean_reward()