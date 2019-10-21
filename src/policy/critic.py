import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from src.common import math_util
from src.common.network_factory import *
from src.common.utils import get_models_dir
import os

model_path = os.path.join(get_models_dir(algorithm="policy"), "critic/")


class Critic(object):
    def __init__(self, obs_dim, args):
        self.args = args
        self.obs_dim = obs_dim  # state dimension
        self.epochs = 10  # optimizer epochs
        self.lr = None  # learning rate set in _build_graph()
        self._build_graph()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')  # observations placeholder
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')  # values placeholder

            self.out, self.lr, _ = build_network(NetworkFactoryInput(
                obs_ph=self.obs_ph,
                obs_dim=self.obs_dim,
                nn_size=self.args.nn_size,
                activation_function=self.args.activation_function,
                type="value",
                structure=self.args.structure,
                hidden_layers=self.args.hidden_layers,
                straight_size=self.args.straight_size,
                kernel_initializer=self.args.kernel_initializer
            ))

            if self.args.learning_rate_value is not None:
                self.lr = self.args.learning_rate_value  # overwrite calculated lr if requested

            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # mse
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.g)
            self.sess.run(self.init)
            if self.args.load:
                print(f"loading critic model")
                self.saver.restore(self.sess, model_path)

    def fit(self, x, y, logger):
        num_batches = self.calc_number_of_batches(x)
        batch_size = x.shape[0] // num_batches

        y_hat = self.predict(x)
        old_variance_explained = math_util.variance_explained(y, y_hat)

        for e in range(self.epochs):
            x_train, y_train = shuffle(x, y)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

        y_hat = self.predict(x)
        loss = math_util.mean_squared_error(y, y_hat)  # calculate total mse
        variance_explained = math_util.variance_explained(y, y_hat)

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': variance_explained,
                    'ExplainedVarDiff': variance_explained - old_variance_explained,
                    'ExplainedVarOld': old_variance_explained})
        return loss

    def calc_number_of_batches(self, x):
        mini_batch_size = self.args.mini_batch_size
        num_batches = max(x.shape[0] // mini_batch_size, 1)
        return num_batches

    def predict(self, x):
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(y_hat)

    def save(self):
        save_path = self.saver.save(self.sess, save_path=model_path)
        print(f"saving critic model to {save_path}")

    def close_sess(self, save=True):
        if save:
            self.save()
        self.sess.close()
