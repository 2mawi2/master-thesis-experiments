import os
from pathlib import Path
from gym.wrappers import FlattenDictWrapper
import random

try:
    import roboschool
except:
    pass

import gym


def get_root_dir():
    # `there must be a better way` R.H.
    root = Path(os.path.abspath(os.path.abspath(__file__))).parent.parent.parent
    return str(root)


def get_logs_dir(algorithm: str):
    return os.path.join(get_root_dir(), "data", algorithm, 'logs')


def get_models_dir(algorithm: str):
    return os.path.join(get_root_dir(), "data", algorithm, 'models')


def get_tensorboard_dir():
    return os.path.join(get_root_dir(), "data", 'tensorboard')


def parse_obs_space(args, env):
    if env.observation_space.shape:
        obs_space = env.observation_space.shape[0]
    else:
        obs_space = env.observation_space.n
    return obs_space


def init_gym(args):
    env = gym.make(args.env_name)
    env.seed(random.randint(1000, 9999))  # random 4 digit seed
    if args.env_name == "FetchReach-v1":  # wrap the FetchReach environment
        env = FlattenDictWrapper(env, ['observation', 'desired_goal', 'achieved_goal'])
    obs_dim = parse_obs_space(args, env)
    if env.action_space.shape:
        act_dim = env.action_space.shape[0]
    else:
        act_dim = env.action_space.n
    return env, obs_dim, act_dim
