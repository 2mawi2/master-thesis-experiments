from gym.wrappers import FlattenDictWrapper

from src.common import segment_generator
from src.common.log_util import log_stats, get_logger
from src.common.utils import init_gym

try:
    import roboschool
except:
    pass

from src.common.kill_handler import KillHandler
from src.common.math_util import calc_disc_sum
from src.common.preprocessor import Preprocessor
from src.policy.actor import Actor
from src.policy.critic import Critic
import numpy as np


def learn(args):
    killer = KillHandler()

    env, obs_dim, act_dim = init_gym(args)
    obs_dim += 1
    logger = get_logger(args)

    preprocessor = Preprocessor(obs_dim, load=args.load)

    critic = Critic(obs_dim, args)
    actor = Actor(obs_dim, act_dim, args)

    segment_generator.generate_segment(env, args, actor, preprocessor,
                                       logger)  # run to find min and max in preprocessor

    episode = 0
    step = 0
    while step < args.num_steps:
        # collect segment
        print("running policy")
        segment, steps = segment_generator.generate_segment(env, args, actor, preprocessor, logger)
        print("learning")
        episode += len(segment)
        step += steps
        add_gae(segment, critic, args.gamma, args.lam)
        observes, actions, advantages, disc_sum_rew = segment_generator.unpack_segment(segment)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)  # center and scale advantages
        log_stats(observes, actions, advantages, logger, disc_sum_rew, episode, step)

        # optimize
        value_loss = critic.fit(observes, disc_sum_rew, logger)  # optimize value function
        actor.update(observes, actions, advantages, value_loss, logger)  # optimize policy gradient

        logger.write()
        if killer.is_manual_kill():
            break

    preprocessor.close()
    logger.close()
    actor.close_sess()
    critic.close_sess()


def add_gae(segment, critic, gamma, lam):
    """Generalized Advantage Estimation, Schulman et al, https://arxiv.org/pdf/1506.02438.pdf """
    for trajectory in segment:  # predict baseline -> calculate advantages
        rewards = trajectory['rewards'] * (1 - gamma)
        trajectory['disc_sum_rew'] = calc_disc_sum(rewards, gamma)
        baseline = trajectory['values'] = critic.predict(trajectory['observes'])
        tds = rewards - baseline + np.append(baseline[1:] * gamma, 0)
        trajectory['advantages'] = calc_disc_sum(tds, gamma * lam)


def evaluate(args):
    env, obs_dim, act_dim = init_gym(args)
    obs_dim += 1
    logger = get_logger(args)

    preprocessor = Preprocessor(obs_dim, load=True)
    critic = Critic(obs_dim, args)
    actor = Actor(obs_dim, act_dim, args)
    args.horizon = 30_000
    args.render = True
    segment_generator.generate_segment(env, args, actor, preprocessor, logger)

    actor.close_sess(save=False)
    critic.close_sess(save=False)


def run(args):
    if not args.render:
        learn(args)  # learn a policy
    else:
        evaluate(args)
