import gymnasium as gym
from bbrl.agents import Agent
import torch
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict
# from stable_baselines3.sac.policies import MultiInputPolicy
from .SAC_load import ReplayBuffer, GaussianActor, flatten_obs
class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action

    
'''
classe pour visualizer les reward pour mieux comprendre ce qui s epasse et eventuellement le modifier
'''
class RewardLogger(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.i = 0  
        self.prev_distance = None

    def reset(self, **kwargs): 
        self.prev_distance=0
        return self.env.reset(**kwargs)
        
    def custom_reward(self,current_distance, old_distance, old_reward, f_t):
        new_reward= old_reward
        if current_distance is not None and old_distance is not None:
            new_reward= (current_distance - old_distance)/10 -0.1 + f_t*10
        else:
            print(f"NONE FOUND: current distance: {current_distance}, old_distance {old_distance}")
        return new_reward
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_distance = info.get('distance', None)
        
        f_t= 1 if done else 0
    
        new_reward = self.custom_reward(current_distance, self.prev_distance, reward, f_t)
        
        self.i += 1
        # if self.i % 1== self.i:
        #     print("--- Step Log ---")
        #     print(f"Step: {self.i}")
        #     print("Old distance: ",current_distance,"\t new distance: ",self.prev_distance)
        #     print(f"Original reward: {reward}")
        #     print(f"New reward: {new_reward}")
        #     print("done:", done)
         
        # if done or truncated:
        #     print("=================")
        #     print(f"Episode ended at step {self.i}")
        #     print(f"Final Original reward: {reward}")
        #     print(f"Final New reward: {new_reward}")
        #     print("ft:", f_t)
            
        self.prev_distance = current_distance  
        return obs, new_reward, done, truncated, info
'''
warpper pour recuperer unqieuemnt les observations continues
'''
# class ContinuousObs(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         continu_space = env.observation_space['continuous']
#         self.observation_space = Box(
#             low=continu_space.low,
#             high=continu_space.high,
#             dtype=continu_space.dtype
#         )

#     def observation(self, obs):
       
#         return {"continuous": obs["continuous"]}

# class ContinuousObs(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         if isinstance(env.observation_space, gym.spaces.Dict):
#             continu_space = env.observation_space['continuous']
#             self.observation_space = Box(
#                 low=continu_space.low,
#                 high=continu_space.high,
#                 dtype=continu_space.dtype
#             )
#         else:
#             self.observation_space = env.observation_space
     
#     def observation(self, obs):
#         return {"continuous": obs["continuous"]}

#     # def observation(self, obs):
#     #     if isinstance(obs, dict) and 'continuous' in obs:
#     #         return obs['continuous']
#     #     return obs


class OnlyContinuousActionsWrapper(gym.ActionWrapper):
    """Exposes only the continuous actions to the policy and zeros out discrete actions."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.discrete_actions = {
            key: value
            for key, value in env.action_space.items()
            if isinstance(value,Discrete)
        }
        self.action_space = Dict(
            {
                key: value
                for key, value in env.action_space.items()
                if isinstance(value, Box)
            }
        )

    def action(self, action):
       
        full_action = {**action}
        for key in self.discrete_actions:
            full_action[key] = 0  # zero out discrete actions
        return full_action


class SamplingActor(Agent):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))




class SB3PPOActor(Agent):
    def __init__(self, observation_space, action_space, state):
        super().__init__()
        
        self.policy = MultiInputPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 0.0,
        )

        # print("action_space", action_space)

        self.policy.load_state_dict(state)
        self.policy.eval()

    def forward(self, t: int):
        obs_continuous = self.get(("env/env_obs/continuous", t))
        obs_dicrete = self.get(("env/env_obs/discrete", t))
        obs = {
            'discrete': obs_discrete,
            'continuous': obs_continous
        }


        obs_tensor, _ = self.policy.obs_to_tensor(obs)
        # print("obs_tensor \n", obs_tensor)
        
        with torch.no_grad():
            dist = self.policy.get_distribution(obs_tensor)
            action = dist.mode()
        
        self.set(("action", t), action)


class SB3SACActor(Agent):
    def __init__(self, observation_space, action_space, state):
        super().__init__()
        
        self.policy = MultiInputPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 0.0,
        )


        self.policy.load_state_dict(state)
        self.policy.eval()

    def forward(self, t: int):
        obs_continuous = self.get(("env/env_obs/continuous", t))
        obs_discrete = self.get(("env/env_obs/discrete", t))
        obs = {
            'discrete': obs_discrete,
            'continuous': obs_continuous
        }


        obs_tensor, _ = self.policy.obs_to_tensor(obs)
        # print("obs_tensor \n", obs_tensor)
        
        with torch.no_grad():
            action, _ = self.policy.predict(obs, deterministic=True)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action).to(obs_continuous.device)
        self.set(("action", t), action)

'''
Garde seulementles actions qui aident a la conduite (ne gere pas fire, nitro et drift)
'''
class basicdrivingAct(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space

    def action(self, action):
        modified_action = action.copy()
        modified_action["fire"] = 0
        modified_action["nitro"] = 0
        modified_action["drift"] = 0   

        return modified_action

'''
Garde seulementles actions qui aident a la conduite (ne gere pas fire, nitro et drift)
'''
class basicdrivingObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        modified_obs = obs.copy()

        driving_obs = {"velocity", "front","distance_down_track", "center_path", "center_path_distance", "max_steer_angle", "jumping", "paths_start","paths_end", "paths_width"}

        for key in obs.keys():
            if key not in driving_obs:
                
                val = obs[key]
                # print(key, val)
                if isinstance(val, np.ndarray):
                    modified_obs[key] = np.zeros_like(val)
                elif isinstance(val, (tuple, list)):
                    modified_obs[key] = tuple(np.zeros_like(v) for v in val)
                else:
                    modified_obs[key] = 0
        
        return modified_obs


class SACInferenceActor(Agent):
    def __init__(self, observation_space, action_space, actor_state, device="cpu"):
        super().__init__()

        self.device = device
        self.observation_space = observation_space
        obs_dim = (
            observation_space.spaces["continuous"].shape[0]
            + sum(observation_space.spaces["discrete"].nvec)
        )
        act_dim = action_space.shape[0]

        self.actor = GaussianActor(obs_dim, act_dim).to(device)
        self.actor.load_state_dict(actor_state)
        self.actor.eval()

        # Action scaling
        self.register_buffer(
            "action_low",
            torch.tensor(action_space.low, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_high",
            torch.tensor(action_space.high, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, t: int):
        # Read observation from BBRL workspace
        obs = {
            "continuous": self.get(("env/env_obs/continuous", t)),
            "discrete": self.get(("env/env_obs/discrete", t)),
        }

        # Flatten EXACTLY like training
        obs_flat = flatten_obs(obs, self.observation_space)
        obs_tensor = (
            torch.from_numpy(obs_flat)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        # Deterministic SAC action
        with torch.no_grad():
            mu, _ = self.actor(obs_tensor)
            action = torch.tanh(mu)

        # Rescale to env bounds
        action = self.action_low + (action + 1.0) * 0.5 * (
            self.action_high - self.action_low
        )

        # Write action to workspace
        self.set(("action", t), action)
