from dataclasses import dataclass
from typing import Any
import tensorflow as tf
from src.common.math_util import geometric_mean
import numpy as np
import os


@dataclass
class NetworkFactoryInput:
    obs_ph: Any
    obs_dim: int
    type: str = "policy"  # policy or value
    act_dim: int = None  # not needed for value network
    activation_function: str = "tanh"
    straight_size: int = 64
    nn_size: int = 10
    hidden_layers: int = 3
    structure: str = "compressing"  # straight or compressing
    kernel_initializer: Any = None  # to overwrite kernel_initializer

    def is_straight(self):
        return self.structure is "straight"

    def is_value(self):
        return self.type is "value"


def build_network(inp: NetworkFactoryInput):
    """
    Network builder for policy based learning
    """
    _validate_input(inp)

    layers = calc_network_structure(inp)

    input = inp.obs_ph
    activation_function = parse_activation_function(inp.activation_function)

    last_layer = inp.obs_dim
    for n, hid_dim in enumerate(layers[:-1]):
        # kernel_initializer = tf.random_normal_initializer(stddev=stddev)
        kernel_initializer = None
        if activation_function is "tanh":
            kernel_initializer = tf.initializers.glorot_normal()
        elif activation_function is "relu":
            kernel_initializer = tf.initializers.he_normal()
        if inp.kernel_initializer is not None:  # overwrite kernel_initializer to naive initialisation
            kernel_initializer = tf.random_uniform_initializer(-(1.0 / np.sqrt(last_layer)),
                                                               (1.0 / np.sqrt(last_layer)))
        input = tf.layers.dense(input, hid_dim, activation_function, name=f"{inp.type}-hidden{n + 1}",
                                kernel_initializer=kernel_initializer)
        last_layer = hid_dim
    kernel_initializer = None
    if activation_function is "tanh":
        kernel_initializer = tf.initializers.glorot_normal()
    elif activation_function is "relu":
        kernel_initializer = tf.initializers.he_normal()
    if inp.kernel_initializer is not None:  # overwrite kernel_initializer to naive initialisation
        kernel_initializer = tf.random_uniform_initializer(-(1.0 / np.sqrt(last_layer)),
                                                           (1.0 / np.sqrt(last_layer)))
    out = tf.layers.dense(input, layers[-1], None, name=f"{inp.type}-out",  # None = linear activation
                          kernel_initializer=kernel_initializer)
    if inp.is_value():
        out = tf.squeeze(out)

    network_depth = get_network_depth(inp, layers)

    lr = (0.0108 if inp.is_value() else 9e-4) / (np.sqrt(network_depth) * 1.5)

    print(f"Intialised {inp.structure} {inp.type} network with layers: "
          f"{', '.join([str(l) for l in layers])}")

    return out, lr, {"layers": layers}


def calc_network_structure(inp: NetworkFactoryInput):
    layers = []
    if inp.is_straight():
        for i in range(inp.hidden_layers):
            layers.append(inp.straight_size)
        layers.append(1 if inp.is_value() else inp.act_dim)  # output layer
    else:
        layers = [0] * inp.hidden_layers
        layers[0] = inp.obs_dim * inp.nn_size

        layers[-1] = np.maximum(inp.nn_size // 2, 5) if inp.is_value() else inp.act_dim * inp.nn_size
        layers = interpolate_with_geometric_mean(layers)
        layers.append(1 if inp.is_value() else inp.act_dim)  # output layer
    return layers


def _validate_input(inp: NetworkFactoryInput):
    if inp.act_dim is None and inp.type is "policy":
        raise ValueError("act_dim must be given to build policy network")
    if inp.structure is "compressing" and inp.hidden_layers % 2 == 0:
        raise ValueError("size of hidden layer must be an odd number")


def get_network_depth(inp: NetworkFactoryInput, layers):
    if inp.is_straight():
        return inp.straight_size
    else:
        center_hidden_layer = layers[(len(layers) - 1) // 2]
        return center_hidden_layer


def parse_activation_function(activation_function: str):
    if activation_function is "relu":
        return tf.nn.relu
    elif activation_function is "tanh":
        return tf.nn.tanh
    else:
        raise NotImplementedError()


def interpolate_with_geometric_mean(layers):
    """
    recursive calculation of the compressing net by interpolating with the geometric mean
    [100,0 ,0 ,0 ,5] -> [100, 46, 22, 10, 5]
    """
    middle = len(layers) // 2
    layers[middle] = geometric_mean(layers[0], layers[-1])
    if len(layers) == 3:
        return layers
    left_half = layers[:middle + 1]
    right_half = layers[middle:]

    if len(left_half) % 2 == 0 or len(right_half) % 2 == 0:
        raise AttributeError("invalid hidden layer size")

    return interpolate_with_geometric_mean(left_half)[:-1] + interpolate_with_geometric_mean(right_half)
