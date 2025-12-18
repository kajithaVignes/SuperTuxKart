from pathlib import Path
from functools import partial
import inspect
import torch
import numpy as np
import gymnasium as gym
import pystk2_gymnasium 

from gymnasium.wrappers import FlattenObservation

from pystk2_gymnasium import AgentSpec
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

from .pystk_actor import env_name, get_wrappers, player_name

from torch.utils.tensorboard import SummaryWriter

from .SAC_load import SACAgent, flatten_obs


    
if __name__ == "__main__":

   
    
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        difficulty=0,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]
    print("Environment name: ", env_name)
    print("Obs space:", env.observation_space)
    print("Act space:", env.action_space)

    if isinstance(env.observation_space, gym.spaces.Dict):
        cont_dim = env.observation_space.spaces["continuous"].shape[0]
        disc_space = env.observation_space.spaces["discrete"]  # MultiDiscrete
        disc_dim = int(disc_space.nvec.sum())
        obs_dim = cont_dim + disc_dim
    else:
        obs_dim = env.observation_space.shape[0]

    act_dim = env.action_space.shape[0]
    
    print("obs_dim", obs_dim)
    print("act_dim", act_dim)

     # TensorBoard writer
    tb_writer = SummaryWriter(log_dir="./tb_logs/SAC_manuel")
    
    agent = SACAgent(obs_dim, act_dim, writer=tb_writer)

   

    total_steps = 1500000
    log_interval = 10000
    start_steps = 30000

    global_step = 0
    episode = 0
    ep_reward = 0.0
    
    obs, _ = env.reset()
    
    while global_step < total_steps:
        obs_flat = flatten_obs(obs, env.observation_space)

        if global_step < start_steps: #warmup SAC
            action = env.action_space.sample() 
        else:
            action = agent.select_action(obs_flat)
        
       
    
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
        next_obs_flat = flatten_obs(next_obs, env.observation_space)
    
        agent.replay_buffer.store(
            obs_flat, action, reward, next_obs_flat, done
        )

        if global_step >= start_steps:
            agent.step()
        
    
        ep_reward += reward
        global_step += 1
        obs = next_obs
    
        if done:
            tb_writer.add_scalar("Reward/Episode", ep_reward, episode)
            tb_writer.add_scalar("Reward/Step", ep_reward, global_step)
    
            print(
                f"Episode {episode} | "
                f"Steps {global_step} | "
                f"Reward {ep_reward:.2f}"
            )
    
            obs, _ = env.reset()
            ep_reward = 0.0
            episode += 1
    
        # ---- optional periodic logging ----
        if global_step % log_interval == 0:
            print(f"[Step {global_step}/{total_steps}]")


    # Save the actor
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(agent.actor.state_dict(), mod_path / "pystk_actor.pth")

    tb_writer.close()
    env.close()

