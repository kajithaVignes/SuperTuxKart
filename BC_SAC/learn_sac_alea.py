
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


if __name__ == "__main__":

    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        num_kart=2,
        agent=AgentSpec(use_ai=False, name=player_name),
    )
    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)
    # actor = Actor(env.observation_space, env.action_space)
    policy_kwargs = dict(
    net_arch=[256, 256],  
    activation_fn=torch.nn.ReLU,
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs/sac_alea"
    )
    

    model.learn(
        total_timesteps=500_000,
        tb_log_name="sac_reward_alea_init",
    )
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(model.policy.state_dict(), mod_path / "pystk_actor_alea.pth")
    print("politique optimale")
    env.close()

    