
from pathlib import Path
from functools import partial
import inspect
import torch
import gymnasium as gym
import pystk2_gymnasium  # IMPORTANT
from stable_baselines3 import SAC
from gymnasium.wrappers import FlattenObservation
import numpy as np

from pystk2_gymnasium import AgentSpec
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

from .pystk_actor import env_name, get_wrappers, player_name
from .actors import extract_driving_obs,extract_continuous_action
from pystk2_gymnasium import ConstantSizedObservations

N_PATHS = 5
    
if __name__=="__main__":
    env = gym.make(
        "supertuxkart/full-v0",
        agent=AgentSpec(use_ai=True),
        num_kart=2,
        render_mode=None,
    )

    env = ConstantSizedObservations(env, state_paths=5)
    print("Obs space:", env.observation_space.shape)
    print("Act space:", env.action_space.shape)
    observations = []
    actions = []
    
    obs, info = env.reset()
    print(obs["action"])
    
    N=10000
    for i in range(N):

        expert_action = obs["action"]

        # on garde le dictionnaire complet on fera le tri après
        obs_vec = extract_driving_obs(obs)
        act_vec = extract_continuous_action(obs["action"])
        #print(act_vec)
        #print("Obs shape:", obs_vec.shape)
        #print("Obs vector:", obs_vec)

        observations.append(obs_vec)
        actions.append(act_vec) 
        obs, reward, terminated, truncated, info = env.step(expert_action)
        if i % 1000 == 0:
            print("tout est ok pour l'instant")
        if terminated or truncated:
            obs, info = env.reset()


    
    data = {
        "obs": torch.tensor(np.array(observations)),
        "actions": torch.tensor(np.array(actions)),
    }

    Path("datasets").mkdir(exist_ok=True)
    torch.save(data, "datasets/bc_dataset.pt")

    env.close()
    

