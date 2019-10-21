import argparse
import os

from src.policy import ppo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser(description='Run policy network implementation')

    add_general_arguments(parser)
    add_policy_arguments(parser)

    return parser.parse_args()


def add_general_arguments(parser):
    parser.add_argument('-a', '--alg', type=str, help='algorithm to run',
                        choices=["ppo"],
                        default="ppo")
    parser.add_argument('-e', '--env_name', type=str,
                        help='OpenAI Gym environment name', default="FetchReach-v1")
    parser.add_argument('-r', '--render', action='store_true', default=False)
    parser.add_argument('-lp', '--logging_path', type=str, help='logging path', default=None)
    parser.add_argument('-n', '--num_steps', type=int, help='number of steps to run', default=1_000_000)
    parser.add_argument('-g', '--gamma', type=float, help='discount factor for value and policy network', default=0.995)


def add_policy_arguments(parser):
    parser.add_argument('-lrp', '--learning_rate_policy', type=float, help='overwrites adaptive policy lr',
                        default=None)
    parser.add_argument('-keri', '--kernel_initializer', type=float,
                        help='std of kernel initializer kernel_initializer', default=None)
    parser.add_argument('-lrv', '--learning_rate_value', type=float, help='overwrites adaptive value lr', default=None)
    parser.add_argument('-lo', '--load', action='store_true')
    parser.add_argument('-pl', '--policy_loss', type=str, help='policy loss',
                        choices=["clipped_kl", "adaptive_kl", "pg", "surrogate"],
                        default="clipped_kl")
    parser.add_argument('-cm', '--complex_model', action='store_true', default=True)
    parser.add_argument('-af', '--activation_function', type=str, help='activation function to use', default="tanh",
                        choices=["tanh", "relu"])
    parser.add_argument('-l', '--lam', type=float, help='GAE lambda', default=0.98)
    parser.add_argument('-vf-c', '--vf_coef', type=float, help='coefficient for value function loss in policy loss',
                        default=0.5)
    parser.add_argument('-k', '--kl_targ', type=float, help='KL target value', default=0.01)
    parser.add_argument('-b', '--rollout_size', type=int, help='segment size, trajectories', default=20)
    parser.add_argument('-mbz', '--mini_batch_size', type=int, help='size of critic mini batch', default=256)
    parser.add_argument('-hor', '--horizon', type=int, help='step size horizon', default=30000)
    parser.add_argument('-nn', '--nn_size', type=int, help='size factor of the neural networks, for '
                                                           'compressing networks only', default=10)
    parser.add_argument('-cr', '--clip_range', type=float, help='the clip range of the surrogate objective',
                        default=0.2)
    parser.add_argument('-pec', '--policy_entropy_coef', type=float, help='coefficient of the policy entropy bonus',
                        default=1.0)
    parser.add_argument('-str', '--structure', type=str, help='structure of the neural network'
                        , choices=["straight", "compressing"], default="compressing")
    parser.add_argument('-hdl', '--hidden_layers', type=int, help='amount of hidden layers', default=3)
    parser.add_argument('-ss', '--straight_size', type=int, help='width of the hidden layers, for straight '

                                                                 'networks only', default=64)


if __name__ == '__main__':
    args = parse_args()

    if args.alg == "ppo":
        ppo.run(args)
    else:
        raise NotImplementedError(f"{args.alg} is not implemented")
