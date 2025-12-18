from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
import torch
import pystk2_gymnasium
# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import MyWrapper, SamplingActor,RewardLogger, basicdrivingAct, basicdrivingObs, OnlyContinuousActionsWrapper, SACInferenceActor

#: The base environment name (you can change that)
env_name = "supertuxkart/full-v0" 

#: Player name (you must change that)
player_name = "Cailloux"


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
<<<<<<< HEAD
        # Example of a custom wrapper
        lambda env: MyWrapper(env, option="1"),
        # lambda env: basicdrivingObs(env),
        # lambda env: basicdrivingAct(env),
        # lambda env: RewardLogger(env),
        lambda env: pystk2_gymnasium.ConstantSizedObservations(env, state_items=5, state_karts=5, state_paths=5),
        lambda env: pystk2_gymnasium.PolarObservations(env),
        lambda env: OnlyContinuousActionsWrapper(env),
        lambda env: pystk2_gymnasium.FlattenerWrapper(env)
=======
        lambda env: DrivingObsWrapper(env),
        lambda env: ContinuousActionWrapper(env)
>>>>>>> 59f7bb027710f83ba6f0fc772330ff3319277e6a
    ]


def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    """Creates a new actor (BBRL agent) that write into `action`

    :param state: The saved `stk_actor/pystk_actor.pth` (if it exists)
    :param observation_space: The environment observation space (with wrappers)
    :param action_space: The environment action space (with wrappers)
    :return: a BBRL agent
    """


    # Returns a dummy actor
    if state is None:
        return SamplingActor(action_space)

    return SACInferenceActor(
        observation_space=observation_space,
        action_space=action_space,
        actor_state=state
    )
   