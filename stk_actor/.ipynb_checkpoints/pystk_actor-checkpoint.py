from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor, SB3ActorContinue,ContinuousObs,RewardLogger,DrivingObsWrapper,ContinuousActionWrapper

#: The base environment name (you can change that)
env_name = "supertuxkart/full-v0"

#: Player name (you must change that)
player_name = "Cailloux"


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    return [
        lambda env: DrivingObsWrapper(env),
        lambda env: ContinuousActionWrapper(env),
        lambda env: RewardLogger(env),
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
    actor = Actor(observation_space, action_space)

    # Returns a dummy actor
    #if state is None:
        #return SamplingActor(action_space)

     #actor.load_state_dict(state)
    #return Agents(actor, ArgmaxActor())
    return SB3ActorContinue(observation_space, action_space, state)
