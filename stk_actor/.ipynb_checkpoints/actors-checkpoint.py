import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper


OBS_DIM = 47
ACTION_DIM = 1
def flatten_sequence(seq):
    return np.concatenate([np.asarray(x).reshape(-1) for x in seq], axis=0)

def pad_paths(seq, K, dim):
    seq = np.array(seq, dtype=np.float32)
    if len(seq) < K:
        # compléter avec des zéros
        pad = np.zeros((K - len(seq), dim), dtype=np.float32)
        seq = np.vstack([seq, pad])
    return seq[:K] 

#on enleve jump
def extract_driving_obs(obs):
    o = []

    o.append(obs["velocity"].reshape(-1))
    o.append(obs["front"].reshape(-1))
    o.append(obs["center_path"].reshape(-1))

    o.append(np.array([obs["distance_down_track"]]).reshape(-1))
    o.append(np.array([obs["center_path_distance"]]).reshape(-1))
    o.append(np.array([obs["max_steer_angle"]]).reshape(-1))

    K = 5
    paths_start = pad_paths(obs["paths_start"], K, 2).flatten()
    paths_end   = pad_paths(obs["paths_end"], K, 2).flatten()
    paths_width = pad_paths(obs["paths_width"], K, 1).flatten()

    o.extend([paths_start, paths_end, paths_width])

    return np.concatenate(o)


def extract_continuous_action(action):
    return np.array([
        action["steer"][0],
    ], dtype=np.float32)


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action

'''
class ContinuousObs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.Dict):
            continu_space = env.observation_space['continuous']
            self.observation_space = Box(
                low=continu_space.low,
                high=continu_space.high,
                dtype=continu_space.dtype
            )
        else:
            self.observation_space = env.observation_space


    def observation(self, obs):
        if isinstance(obs, dict) and 'continuous' in obs:
            return obs['continuous']
        return obs
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


class DrivingObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Dimension exacte = celle de extract_driving_obs
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32
        )

    def _obs_dim(self):
        # calcule une fois à partir d'un reset
        obs, _ = self.env.reset()
        return extract_driving_obs(obs).shape[0]

    def observation(self, obs):
        
        obs_vec = extract_driving_obs(obs)
        return  obs_vec
        
        #return {
         #  "continuous": extract_driving_obs(obs).astype(np.float32)
        #}
        

            

class ContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # On ne garde que le steer
        self.action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        # On ne met que steer dans le dictionnaire d'action
        return {
            "steer": np.array([action[0]], dtype=np.float32),
            "acceleration": np.array([1.0], dtype=np.float32),  # fixe une valeur si nécessaire
            "brake": 0,
            "drift": 0,
            "fire": 0,
            "nitro": 0,
            "rescue": 0,
        }

class Actor(Agent):
    """Computes probabilities over action"""

    def forward(self, t: int):
        # Computes probabilities over actions
        raise NotImplementedError()


class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        raise NotImplementedError()


class SamplingActor(Agent):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))


class SubmissionActor(Agent):
    def __init__(self, state):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Tanh()
        )

        if state is not None:
            actor_state = {}

            # mapping SB3 → Sequential
            actor_state["0.weight"] = state["actor.latent_pi.0.weight"]
            actor_state["0.bias"]   = state["actor.latent_pi.0.bias"]

            actor_state["2.weight"] = state["actor.latent_pi.2.weight"]
            actor_state["2.bias"]   = state["actor.latent_pi.2.bias"]

            actor_state["4.weight"] = state["actor.mu.weight"]
            actor_state["4.bias"]   = state["actor.mu.bias"]

            self.model.load_state_dict(actor_state)

        self.model.eval()

    def forward(self, t: int):
        obs = self.get(("env/env_obs", t))

        if isinstance(obs, dict):
            obs = extract_driving_obs(obs)
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
    
        with torch.no_grad():
            action = self.model(obs)
    
        self.set(("action", t), action)
