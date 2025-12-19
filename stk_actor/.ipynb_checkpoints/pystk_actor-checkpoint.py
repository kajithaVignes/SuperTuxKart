from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor, SB3ActorContinue,RewardLogger,DrivingObsWrapper,ContinuousActionWrapper,CleanBBRLActor

#: The base environment name (you can change that)
env_name = "supertuxkart/full-v0"

#: Player name (you must change that)
player_name = "Cailloux"


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    return [
        lambda env: DrivingObsWrapper(env),
        lambda env: ContinuousActionWrapper(env)
    ]


def get_actor(state, observation_space, action_space):
    return SB3ActorContinue(
        observation_space=observation_space,
        action_space=action_space,
        state=state,
    )
    
