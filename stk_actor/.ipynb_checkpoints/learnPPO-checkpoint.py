
from pathlib import Path
from functools import partial
import inspect
import torch
import gymnasium as gym
import pystk2_gymnasium  # IMPORTANT
from stable_baselines3 import PPO
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

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_logs/ppo_bc"
    )
    
    
    bc_path = Path("./stk_actor/bc_model.pth")
    bc_state = torch.load(bc_path, map_location="cpu")

    sac_actor = model.policy

    with torch.no_grad():
        for sac_param, bc_param in zip(
            sac_actor.parameters(),
            bc_state.values(),
        ):
            sac_param.copy_(bc_param)

    print("policy initial depuis BC")

    model.learn(
        total_timesteps=500_000,
        tb_log_name="ppo_reward_bc_init",
    )
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(model.policy.state_dict(), mod_path / "pystk_actor_ppo_bc.pth")
    print("politique optimale")
    env.close()

    