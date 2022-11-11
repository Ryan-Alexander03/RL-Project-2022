import os
import gym

from nle import nethack
from minihack import RewardManager

from rewardfn import reward_fn, reward_fn_room

"""
Setup environment
"""
def make_env():

    # Reduced action space
    CUS_ACTION_SPACE = (nethack.CompassDirection.N,nethack.CompassDirection.E,nethack.CompassDirection.S,nethack.CompassDirection.W)

    # Reduced observation space
    CUS_OBSERVATION_SPACE = ( "pixel", "glyphs_crop", "chars")

    # custom reward function
    CUS_REWARD_MANAGER = RewardManager()
    # note you can have multiple custom reward fn
    CUS_REWARD_MANAGER.add_custom_reward_fn(reward_fn) 
    CUS_REWARD_MANAGER.add_custom_reward_fn(reward_fn_room)
    # need this else game crash after 1 timestep
    CUS_REWARD_MANAGER.add_kill_event("minotaur", reward=2.0, terminal_required=True) 

    # initialise and make the environment
    env_id = "MiniHack-Quest-Hard-v0"
    env = gym.make(env_id, observation_keys=CUS_OBSERVATION_SPACE, actions=CUS_ACTION_SPACE, reward_manager=CUS_REWARD_MANAGER)
    
    return env