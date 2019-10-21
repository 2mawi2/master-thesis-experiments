import tensorflow as tf

from src.common import math_util
from src.common.network_factory import build_network, NetworkFactoryInput
from src.common.utils import *
import numpy as np

model_path = os.path.join(get_models_dir(algorithm="policy"), "actor/")


class Actor(object):
    def __init__(self, obs_dim, act_dim, args):
        self.args = args
        self.beta = 1.0  # adaptive penalty coefficient
        self.epochs = 20  # optimizer epochs
        self.nu = 2
        self.omega = 1.5
        self.rho = 1.0  # step size multiplier
        self.rho_min = 0.1  # min value step size multiplier
        self.rho_max = 10  # max value step size multiplier
        self.lr_initial = None  # set based on nn size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy_logvar = -1  # -1 -> natural log of initial policy variance
        self._build_graph()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._build_network()
            self._calc_log_probabilities()
            self._calc_kl_and_entropy()
            self._sample()
            self._build_loss()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.g)
            self.sess.run(self.init)
            if self.args.load:
                print(f"loading actor model")
                self.saver.restore(self.sess, model_path)

    def _placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.value_loss_ph = tf.placeholder(tf.float32, (), 'value_loss')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_out_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_out')

    def _build_network(self):
        self.out, self.lr_initial, info = build_network(NetworkFactoryInput(
            obs_ph=self.obs_ph,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            nn_size=self.args.nn_size,
            activation_function=self.args.activation_function,
            type="policy",
            structure=self.args.structure,
            hidden_layers=self.args.hidden_layers,
            straight_size=self.args.straight_size,
            kernel_initializer=self.args.kernel_initializer
        ))

        if self.args.learning_rate_policy is not None:
            self.lr_initial = self.args.learning_rate_policy

        logvar_speed = (10 * info["layers"][-2]) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32, tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        print(f'Policy Params --, lr: {self.lr_initial:.3g}, logvar_speed: {logvar_speed}')

    def _calc_log_probabilities(self):
        self.logp = math_util.log_prop(self.act_ph, self.out, self.log_vars)
        self.logp_old = math_util.log_prop(self.act_ph, self.old_out_ph, self.old_log_vars_ph)

    def _calc_kl_and_entropy(self):
        self.kl = math_util.calc_kl_divergence(self.old_log_vars_ph, self.old_out_ph,
                                               self.log_vars, self.out,
                                               self.act_dim)

        self.entropy = math_util.calc_entropy(self.act_dim, self.log_vars)

    def _sample(self):
        self.sampled_act = (self.out + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.act_dim,)))

    def _build_adaptive_loss(self):
        print('setting up loss with Adaptive KL Penalty Coefficient')
        pg_loss = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.logp_old))
        penalty_loss = tf.reduce_mean(self.beta_ph * self.kl)
        return pg_loss + penalty_loss

    def _build_clipping_loss(self):
        print('setting up loss with Clipped Surrogate Objective')
        pg_ratio = tf.exp(self.logp - self.logp_old)
        entropy_bonus = tf.reduce_mean(self.entropy)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
        min_surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio, self.advantages_ph * clipped_pg_ratio)
        clipping_loss = -tf.reduce_mean(min_surrogate_loss)
        return clipping_loss - self.args.policy_entropy_coef * entropy_bonus + self.args.vf_coef * self.value_loss_ph

    def _build_surrogate_loss(self):
        print('setting up loss with Surrogate Objective')
        return -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.logp_old))

    def _build_policy_gradient_loss(self):
        print('setting up loss with Vanilla Policy Gradient Objective')
        return -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp))

    def _build_loss(self):
        with tf.name_scope("loss"):
            if self.args.policy_loss == "adaptive_kl":
                self.loss = self._build_adaptive_loss()
            elif self.args.policy_loss == "clipped_kl":
                self.loss = self._build_clipping_loss()
            elif self.args.policy_loss == "surrogate":
                self.loss = self._build_surrogate_loss()
            elif self.args.policy_loss == "pg":
                self.loss = self._build_policy_gradient_loss()
            else:
                raise ValueError("loss unknown")
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)  # s -> pi -> a

    def adjust_beta(self, kl):
        """
        Adjust beta based on KL target
        Schulman et al, https://arxiv.org/pdf/1707.06347.pdf
        """
        if kl > self.args.kl_targ * self.nu:
            self.beta = np.minimum(35, self.omega * self.beta)
        elif kl < self.args.kl_targ / self.nu:
            self.beta = np.maximum(1 / 35, self.beta / self.omega)

    def adjust_rho(self, kl):
        if kl > self.args.kl_targ * self.nu and self.beta > 30 and self.rho > self.rho_min:
            self.rho /= self.omega  # kl is far too large -> reduce Adam step size
        elif kl < self.args.kl_targ / self.nu and self.beta < (1 / 30) and self.rho < self.rho_max:
            self.rho *= self.omega  # kl is far too low -> increase Adam step size

    def update(self, observes, actions, advantages, value_loss, logger):
        lr = self.lr_initial * self.rho  # calculate the adaptive learning rate before each optimization

        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.lr_ph: lr,
                     self.value_loss_ph: value_loss}

        old_outs_np, old_log_vars_np = self.sess.run([self.out, self.log_vars], feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_out_ph] = old_outs_np
        loss, kl, entropy = 0, 0, 0

        # optimization
        for e in range(self.epochs):
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.args.kl_targ * 4:  # early stopping Heess et al 2017 https://arxiv.org/pdf/1707.02286.pdf
                break

        self.adjust_beta(kl)  # adapt penalty coefficient
        if self.args.learning_rate_policy is None:  # if learning rate not fixed
            self.adjust_rho(kl)  # adapt learning rate multiplier

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    "lr_multiplier": self.rho,
                    "lr": self.lr_initial,
                    'Beta': self.beta})

    def save(self):
        save_path = self.saver.save(self.sess, save_path=model_path)
        print(f"saving actor model to {save_path}")

    def close_sess(self, save=True):
        if save:
            self.save()
        self.sess.close()
