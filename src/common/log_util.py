from src.common.kill_handler import KillHandler
from src.common.logger import Logger
from src.common.math_util import calc_disc_sum
from src.common.preprocessor import Preprocessor
from src.policy.actor import Actor
from src.policy.critic import Critic
import numpy as np


def log_stats(observes, actions, advantages, logger, disc_sum_rew, episode, step):
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                'step': step,
                'episode': episode
                })


def get_logger(args) -> Logger:
    return Logger(args, scalars=[
        ("Mean Reward", "MeanReward"),
        ("Batch Length", "steps"),
        ("Critic Loss", "ValFuncLoss"),
        ("Actor Loss", "PolicyLoss"),
        ("Policy Entropy", "PolicyEntropy"),
        ("KL-Divergence", "KL"),
        ("Rolling Mean Reward", "RollingMeanReward"),
        ("Rolling Traj length", "Rollingsteps"),
        ("ExplainedVarNew", "ExplainedVarNew"),
        ("ExplainedVarOld", "ExplainedVarOld"),
        ("ExplainedVarDiff", "ExplainedVarDiff"),
    ], algorithm="policy")
