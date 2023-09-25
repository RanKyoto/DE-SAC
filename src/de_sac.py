from inspect import Parameter
import sys
from turtle import forward
from linear_sac import LogCallBack
sys.path.append('./')
from utils import process_bar
from utils.sensors import ForwardKinematicsGaussianSensor
import gymnasium as gym
from gymnasium import spaces
import myenv

# DENSITY_ESTIMATION = {"KDE", "MAF"}
DENSITY_ESTIMATION = "KDE"
import numpy as np
from typing import Dict, List, Optional, Tuple, Type, Union, Any
import torch as th
from torch import nn
from src.kde import KernelDensityEstimation
from src.maf import MAF_Model


from stable_baselines3.common.type_aliases import  Schedule
from stable_baselines3.sac.policies import SACPolicy,LOG_STD_MAX,LOG_STD_MIN
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy,BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor,FlattenExtractor,create_mlp
from stable_baselines3.common.distributions import DiagGaussianDistribution,SquashedDiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.callbacks import BaseCallback


class DE_SACPolicy(SACPolicy):
    def __init__(
        self, 
        observation_space: spaces.Space,
        action_space: spaces.Space, 
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = th.nn.ReLU,
        use_sde: bool = False, 
        log_std_init: float = -3,
        use_expln: bool = False, 
        clip_mean: float = 2,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True, 
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2, 
        share_features_extractor: bool = False
   
        ):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, use_sde, log_std_init, use_expln, clip_mean, features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class, optimizer_kwargs, n_critics, share_features_extractor)
        self._squash_output = True

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["net_arch"]=[64,64]
        return DEActor(**actor_kwargs).to(self.device)

class DEActor(BasePolicy):
    def __init__(self,       
        observation_space: spaces.Space,
        action_space: spaces.Space, 
        features_extractor: th.nn.Module,
        features_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        squash_output: bool = True,
        **kwargs
        ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        action_dim = get_action_dim(self.action_space)
        self.sensor = ForwardKinematicsGaussianSensor()
        output_dim = self.sensor.output_dim
        actor_net = create_mlp(output_dim, action_dim, net_arch, 
                            activation_fn, squash_output=squash_output)
        self.actor_net = nn.Sequential(*actor_net)
        if DENSITY_ESTIMATION == "KDE":
            self.kde = KernelDensityEstimation(delta=0.2,u_dim=action_dim)
        elif DENSITY_ESTIMATION == "MAF":
            self.maf = MAF_Model(input_dim= action_dim, 
                label_dim= features_dim) 
        elif DENSITY_ESTIMATION == "TEST":
            pass
        else:
            assert False, "DENSITY_ESTIMATION should be KDE or MAF"

    def get_actor_parameters_vector(self)->np.ndarray:
        return  th.nn.utils.parameters_to_vector(self.get_submodule("actor_net").parameters()).detach().cpu().numpy()

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if DENSITY_ESTIMATION is not None:
            with th.no_grad():
                y = self.sensor(obs).squeeze()
            return self.actor_net(y)
        else:
            assert False, "DENSITY_ESTIMATION should be KDE or MAF"


    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if DENSITY_ESTIMATION == "KDE":
            with th.no_grad():
                y = self.sensor(obs) #sim dim is not needed
                y_data = self.sensor.sensor_sim(obs,sim_num=500)
            
            actions = self.actor_net(y)            
            u_data = self.actor_net(y_data)

            log_pdf = self.kde.log_pdf(u=actions,u_data= u_data)
            return actions.squeeze(), log_pdf
        elif DENSITY_ESTIMATION == 'MAF':
            with th.no_grad():
                y = self.sensor(obs).squeeze(dim=1) #sim dim is not needed
                 
            actions = self.actor_net(y)
            log_pdf = self.maf.log_prob(actions,obs)

            return actions, log_pdf 
        elif DENSITY_ESTIMATION == 'TEST':
            with th.no_grad():
                y = self.sensor(obs).squeeze(dim=1) #sim dim is not needed
                 
            actions = self.actor_net(y)
            return actions,th.zeros((len(obs),),device=self.device)
        else:
            assert False, "DENSITY_ESTIMATION should be KDE or MAF"

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

class DE_SAC():
    class MAFCallBack(BaseCallback):
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            if DENSITY_ESTIMATION == "MAF":
                batch_size,sim_num= 16, 512
                dim = self.model.actor.features_dim
                loss_list = []
                for i in range(10):
                    with th.no_grad():    
                        x_data = (th.rand((batch_size,dim),device=self.model.device)-0.5)*4
                        y_data = self.model.actor.sensor.sensor_sim(x_data,sim_num = sim_num)
                        u_data = self.model.actor.actor_net(y_data).reshape((-1,dim)) 
                        labels = x_data.repeat((sim_num,1,1)).transpose(0,1).reshape((-1,dim))
                    loss = self.model.actor.maf.learn(u_data,labels)
                    loss_list.append(loss.item())
                mean_loss  = np.mean(loss_list)    
                    #print(np.mean(loss_list))
                self.model.logger.record("train/MAF_loss", mean_loss)
            return True

    def __init__(self,loadpath=None,total_timesteps:int= 100000,render_mode ="rgb_array") -> None:
        prefix = './data/'
        self.env = gym.make("MechArm-v0",render_mode=render_mode)
        if loadpath is None or loadpath=='':
            loadpath = 'de_sac'
            if DENSITY_ESTIMATION is not None:
                loadpath += '_'+DENSITY_ESTIMATION
            logpath = prefix+loadpath+'_tensorboard/' #tensorboard_log = logpath,
            model =  SAC(DE_SACPolicy, self.env, verbose=1,gamma=0.99,tensorboard_log = logpath,
                     buffer_size=int(10e4),batch_size=256,learning_rate=0.001,ent_coef=0.001)
               
            model.learn(total_timesteps,log_interval=1,callback=self.MAFCallBack(),progress_bar=True)
            model.save(prefix+loadpath)
            del model
        model =  SAC(DE_SACPolicy, self.env, verbose=1,gamma=0.9,
             buffer_size=int(10e4),batch_size=256,learning_rate=0.000,ent_coef=0.001)
        self.model = model.load(prefix+loadpath)

    def simulation(self):
        n=100
        for ep in range(n):
            state,_ = self.env.reset(options={"random":True}) 
            for i in range(600):
                with th.no_grad():
                    output = self.model.actor.sensor(th.tensor(state,device=agent.model.device).unsqueeze(0)).flatten()
                    action = self.model.actor.actor_net(output).tolist()
                state,_,_,_,_ = self.env.step(action=action)

    def plot(self):
        n=1
        y0 = np.array([1.68000004e-01, 3.85686529e-07, 2.43000000e-01])
        norm_list = []
        noisy_output_list = []
        output_list = []
        input_list = []
        norm_list1 = []
        for ep in range(n):
            state,_ = self.env.reset(options={"random":False,"x0":[0.5,-1,-0.5,0.5,0.5,0]}) 
            for i in range(1000):
                with th.no_grad():
                    output = self.model.actor.sensor(th.tensor(state,device=agent.model.device).unsqueeze(0)).flatten()
                    #output = th.tensor(self.env.output[0:3],device=agent.model.device,dtype=th.float32)
                    action = self.model.actor.actor_net(output).tolist()
                    norm_list1.append(np.linalg.norm(np.array(output.tolist())-self.env.output[0:3]))
                    norm_list.append(np.linalg.norm(y0-self.env.output[0:3]))
                    noisy_output_list.append(np.linalg.norm(output.tolist()))
                    output_list.append(np.linalg.norm(self.env.output[0:3]))
                    input_list.append(np.linalg.norm(action))


                state,_,_,_,_ = self.env.step(action=action)

        import matplotlib.pyplot as plt
        plt.rc("text",usetex=True)
        fig,axes = plt.subplots(3,1,sharex="all")
        axes[0].plot(norm_list,color="blue",label=r"$\Vert g_{\rm FK6}(x_k) -g_{\rm FK6}(\mathbf{0}_6)\Vert_2$")
        axes[0].plot(norm_list1,color="red",alpha=0.8,linestyle=":",label=r"$\Vert y_k -g_{\rm FK6}(\mathbf{0}_6)\Vert_2$")
        axes[0].set_ylabel("distance to the target")
        axes[0].legend(loc = "upper right")
        axes[1].plot(output_list,color="blue",label=r"$\Vert g_{\rm FK6}(x_k) \Vert_2$")
        axes[1].plot(noisy_output_list,color="red",alpha=0.8,linestyle=":",label=r"$\Vert y_k \Vert_2$")
        axes[1].set_ylabel("outputs")
        axes[1].legend(loc = "upper right")
        axes[2].plot(input_list,color = "blue", label=r"$\Vert u_k \Vert_2$")
        axes[2].set_ylabel("inputs")
        axes[2].legend(loc = "upper right")
        plt.xlim([-10,1000])
        plt.xlabel(r"$k$")
        plt.tight_layout()
        plt.savefig("./plot/sim3_kde.pdf")


if __name__ == '__main__':   
    DENSITY_ESTIMATION = "KDE"
    agent = DE_SAC("de_sac_KDE",total_timesteps=100000,render_mode='rgb_array')
    agent.plot()