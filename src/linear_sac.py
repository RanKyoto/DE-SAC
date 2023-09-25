import sys
sys.path.append('./')
from utils import process_bar,LinearGaussianSensor
import gymnasium as gym
from gymnasium import spaces
import myenv

env = gym.make("Linear-v0",render_mode='rgb_array')
ENV_C = env.C
ENV_G = env.G

# DENSITY_ESTIMATION = {None, "KDE", "MAF"}
DENSITY_ESTIMATION = "KDE"
import numpy as np
from typing import Dict, List, Optional, Tuple, Type, Union, Any
import torch as th
from torch import nn
from src.kde import KernelDensityEstimation1D
from src.maf import LinearGaussainMAF


from stable_baselines3.common.type_aliases import  Schedule
from stable_baselines3.sac.policies import SACPolicy,LOG_STD_MAX,LOG_STD_MIN
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor,FlattenExtractor,create_mlp
from stable_baselines3.common.distributions import DiagGaussianDistribution,SquashedDiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.callbacks import BaseCallback

class LinearSACPolicy(SACPolicy):
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
        return LinearGaussianActor(**actor_kwargs).to(self.device)

class LinearGaussianActor(BasePolicy):
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

        self.Kin = nn.Parameter(th.randn(5))
        K_net = create_mlp(input_dim=5,
                output_dim=-1,
                net_arch=net_arch,
                activation_fn=activation_fn)
        self.K_latent=nn.Sequential(*K_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.K_net = nn.Linear(last_layer_dim,action_dim)
              
        self.sensor = LinearGaussianSensor(C = ENV_C, G= ENV_G)
        self.kde = KernelDensityEstimation1D()
        self.maf = LinearGaussainMAF()
        self.maf.load()

        if squash_output:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        else:
            self.action_dist = DiagGaussianDistribution(action_dim)
  
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if DENSITY_ESTIMATION is None:
            mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
            return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
        else:
            y = self.sensor(obs).squeeze(dim=1)
            return th.tanh(self.K @ y)
 

    @property
    def K(self)->th.Tensor:
        K_latent = self.K_latent(self.Kin)
        return self.K_net(K_latent)
    
    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        K = self.K
        mean_actions =  K * features.matmul(self.sensor.C.to(self.device).T)
        log_std = (K*float(self.sensor.G)+0.00001).abs().log()
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean_actions, log_std, {}

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        if DENSITY_ESTIMATION is None:
            mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
            # return action and associated log prob
            return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
        elif DENSITY_ESTIMATION == "KDE":
            with th.no_grad():
                y = self.sensor(obs).squeeze(dim=1) #sim dim is not needed
                y_data = self.sensor.sensor_sim(obs,sim_num=10000)
            
            mean_actions = th.tanh(self.K @ y)            
            u_data = th.tanh(self.K @ y_data)
            log_pdf,_ = self.kde.log_pdf(u=mean_actions,u_data= u_data)
            
            return mean_actions, log_pdf
        elif DENSITY_ESTIMATION == 'MAF':
            with th.no_grad():
                y = self.sensor(obs).squeeze(dim=1) #sim dim is not needed
            mean_actions = th.tanh(self.K @ y)
            labels = th.hstack((obs,self.K.repeat(len(obs),1)))  
            log_pdf = self.maf.model.log_prob(mean_actions,labels)
            return mean_actions,log_pdf
        elif DENSITY_ESTIMATION == "TEST":
            with th.no_grad():
                y = self.sensor(obs).squeeze(dim=1) #sim dim is not needed
            mean_actions = th.tanh(self.K @ y)
            return mean_actions, th.zeros((len(obs),),device=self.device)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

class LogCallBack(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        with th.no_grad():
            K = self.model.actor.K
        self.model.logger.record("train/K", K.item())
        return True

class LinearSAC():
    def __init__(self,loadpath=None,total_timesteps:int= 100000) -> None:
        prefix = './data/'
        if loadpath is None or loadpath=='':
            loadpath = 'linear_sac'
            if DENSITY_ESTIMATION is not None:
                loadpath += '_'+DENSITY_ESTIMATION
            logpath = prefix+loadpath+'_tensorboard/'
            log_call = LogCallBack()
            model =  SAC(LinearSACPolicy, env, verbose=1,gamma=0.9,tensorboard_log = logpath,
                     buffer_size=int(10e4),batch_size=256,learning_rate=0.0001,ent_coef=0.001)
            model.learn(total_timesteps,log_interval=1,callback=log_call,progress_bar=True)
            model.save(prefix+loadpath)
            del model
        model =  SAC(LinearSACPolicy, env, verbose=1,gamma=0.9,
             buffer_size=int(10e4),batch_size=256,learning_rate=0.0001,ent_coef=0.001)
        self.model = model.load(prefix+loadpath)

    def simulation(self,n_step:int = 300)->float:
        ep_reward = 0.0
        env.reset()
        for _ in range(n_step):
            with th.no_grad():
                K = self.model.actor.K
            action = np.tanh(env.output * K.item())*env.u_mag
            _,reward,_,_,_ = env.step(action)
            ep_reward += reward
        return ep_reward

    def mean_reward(self,n_ep:int =1000):
        reward_list = []
        for ep in range(n_ep):
            mean_reward = self.simulation(n_step=300)
            reward_list.append(mean_reward)
            process_bar((ep+1)/n_ep,end_str="|reward={}".format(np.mean(reward_list)))
        return np.mean(reward_list)
        
# testing code
if __name__ == '__main__':   
    DENSITY_ESTIMATION = "TEST"
    sac = LinearSAC('',total_timesteps=20000)
    #sac.mean_reward()

