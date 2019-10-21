import scipy.signal
import tensorflow as tf
import numpy as np


def calc_disc_sum(rewards: [], gamma: float) -> []:
    """
    Calculates discounted sum of rewards
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]


def geometric_mean(first: int, second: int) -> int:
    return int(np.sqrt(first * second))


def variance_explained(y, y_hat):
    return 1 - np.var(y_hat - y) / np.var(y)


def mean_squared_error(y, y_hat):
    return np.mean(np.square(y_hat - y))


def log_prop(act_ph, means, log_vars):
    probs = -0.5 * tf.reduce_sum(log_vars)
    probs += -0.5 * tf.reduce_sum(tf.square(act_ph - means) / tf.exp(log_vars), axis=1)
    return probs


def calc_entropy(act_dim, log_vars):
    return 0.5 * (act_dim * (np.log(2 * np.pi) + 1) + tf.reduce_sum(log_vars))


def calc_kl_divergence(old_log_vars, old_means, log_vars, means, act_dim):
    """see http://web.stanford.edu/~jduchi/projects/general_notes.pdf p.13"""
    log_det_cov_old = tf.reduce_sum(old_log_vars)
    log_det_cov_new = tf.reduce_sum(log_vars)
    tr_old_new = tf.reduce_sum(tf.exp(old_log_vars - log_vars))
    kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                              tf.reduce_sum(tf.square(means - old_means) / tf.exp(log_vars),
                                            axis=1) - act_dim)
    return kl
