import os
import shutil

from multiprocessing import Process
import time

from tensorboard.compat import tf

from src.common.utils import get_root_dir, get_tensorboard_dir
from src.policy import ppo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def evaluate(target, args, runs: int = 1, parallel_runs: int = 4):
    shutil.rmtree(os.path.join(get_tensorboard_dir()), ignore_errors=True)  # clear tensorboard dir first
    for run in range(runs):
        processes = []
        for i in range(parallel_runs):
            p = Process(target=target, args=(args,))
            time.sleep(2)  # wait one second to avoid log interference
            p.start()
            processes.append(p)
        [i.join() for i in processes]
    print("evaluation done...")


class Expando:
    pass


def get_default_args():
    args = Expando()
    args.load = False
    args.render = False
    args.adaptive_kl = False
    args.policy_loss = "clipped_kl"
    args.learning_rate_policy = None
    args.learning_rate_value = None
    args.num_steps = 10_000
    args.gamma = 0.995
    args.lam = 0.98
    args.kl_targ = 0.01
    args.rollout_size = 20
    args.nn_size = 10
    args.clip_range = 0.2
    args.activation_function = "tanh"
    args.structure = "compressing"
    args.hidden_layers = 3
    args.straight_size = 64
    args.policy_entropy_coef = 0.0
    args.vf_coef = 0.0
    args.mini_batch_size = 256
    args.horizon = 30_000
    args.kernel_initializer = None
    return args


def run_evaluation(args, env, experiment, steps=500_000, algorithm="policy", parallel_runs=4, runs=1):
    print(f"starting experiment: {experiment}, "
          f"for environment: {env} steps: {steps}, parallel_runs: {parallel_runs}, runs: {runs}")
    args.logging_path = os.path.join(get_root_dir(), "data", algorithm, 'results', experiment)
    args.env_name = env
    args.num_steps = steps
    evaluate(ppo.run, args, runs=runs, parallel_runs=parallel_runs)


def simple_vs_complex():
    args = get_default_args()
    # run_evaluation(args=args, env="InvertedDoublePendulum-v2", experiment=f"ppo-clipped-complex-5-lr")

    run_evaluation(args=args, env="InvertedDoublePendulum-v2", experiment=f"ppo-clipped-simple-1")


def relu_exp():
    args = get_default_args()
    args.activation_function = "relu"
    run_evaluation(args=args, env="InvertedDoublePendulum-v2", experiment=f"ppo-clipped-simple-relu")

    args.activation_function = "relu"
    run_evaluation(args=args, env="InvertedDoublePendulum-v2", experiment=f"ppo-clipped-complex-relu")


def pg():
    args = get_default_args()
    args.activation_function = "tanh"
    run_evaluation(args=args, env="InvertedDoublePendulum-v2", experiment=f"pg-complex-tanh")


def ppo_clipping(clip_range=0.2):
    args = get_default_args()
    args.clip_range = clip_range
    args.activation_function = "tanh"
    run_evaluation(args=args, env="InvertedDoublePendulum-v2", experiment=f"pg-complex-clp-{clip_range}")


def test():
    args = get_default_args()
    # run_evaluation(args=args, env="InvertedDoublePendulum-v2", experiment=f"ppo-clipped-complex-5-lr")
    run_evaluation(args=args, env="Hopper-v2", experiment=f"test", steps=100_000, parallel_runs=1, runs=1)


def nn_size_comparasion():
    args = get_default_args()
    args.structure = "straight"
    args.hidden_layers = 4
    args.straight_size = 64
    run_evaluation(args=args, env="InvertedDoublePendulum-v2",
                   experiment=f"ppo-clipped-nn_size-50", steps=500_000)


def losses():
    args = get_default_args()
    run_evaluation(args=args, env="InvertedDoublePendulum-v2",
                   experiment=f"ppo-clipped-nn_size-losses-clipped", steps=500_000)


def rollout_size(sizes):
    for size in sizes:
        args = get_default_args()
        args.rollout_size = size
        run_evaluation(args=args, env="InvertedDoublePendulum-v2",
                       experiment=f"ppo-clipped-rollout-size-{size}", steps=500_000)


def policy_entropy_bonus(coeffs):
    for coef in coeffs:
        args = get_default_args()
        args.policy_entropy_coef = coef
        run_evaluation(args=args, env="Hopper-v2",
                       experiment=f"ppo-network-factory-value-function-loss", steps=1_000_000, parallel_runs=4)


def min_batch_size(mini_batch_sizes):
    for mbz in mini_batch_sizes:
        args = get_default_args()
        args.mini_batch_size = mbz
        run_evaluation(args=args, env="Hopper-v2",
                       experiment=f"ppo-min-batch-size-{mbz}", steps=1_000_000, parallel_runs=4)


def no_dynamic_step_size():
    args = get_default_args()
    run_evaluation(args=args, env="Hopper-v2",
                   experiment=f"ppo-dynamic-step-64-64-straight", steps=1_000_000, parallel_runs=4)


def value_function_coef(coeffs):
    for coeff in coeffs:
        args = get_default_args()
        args.vf_coef = coeff
        run_evaluation(args=args, env="Hopper-v2",
                       experiment=f"value_function_coef-{coeff}", steps=1_000_000, parallel_runs=4)


def policy_entropy_coef(coeffs):
    for coeff in coeffs:
        args = get_default_args()
        args.policy_entropy_coef = coeff
        run_evaluation(args=args, env="Hopper-v2",
                       experiment=f"policy_entropy_coef-{coeff}", steps=1_000_000, parallel_runs=4)


def kl_targ(coeffs):
    for coeff in coeffs:
        args = get_default_args()
        args.kl_targ = coeff
        run_evaluation(args=args, env="Hopper-v2",
                       experiment=f"kl_targ-{coeff}", steps=1_000_000, parallel_runs=4)


def nn_size_comp(fixed_lr=False, sizes=range(1, 30)):
    environment = "Hopper-v2"
    for nn_size in sizes:
        args = get_default_args()
        args.nn_size = nn_size
        if fixed_lr:  # no adaptive learning rate
            args.learning_rate_value = 0.0014696938456699071
            args.learning_rate_policy = 7.745966692414833e-05
        run_evaluation(args=args, env=environment,
                       experiment=f"ppo-{'fixed' if fixed_lr else 'adaptive'}-{nn_size}-new", steps=1_000_000,
                       parallel_runs=6, runs=1)


def nn_size_comp_adaptive(fixed_lr=False, sizes=range(1, 30)):
    environment = "Hopper-v2"
    for nn_size in sizes:
        args = get_default_args()
        args.nn_size = nn_size
        args.policy_loss = "adaptive_kl"
        if fixed_lr:  # no adaptive learning rate
            # # learning rate of adaptive 1
            # critic 0.0027213442056664362
            # actor 0.00024494897427831784
            # learning rate of adaptive 10
            # actor 7.745966692414833e-05
            # critic 0.0014696938456699071
            # learning rate of adaptive 20
            # actor 5.4772255750516614e-05
            # critic 0.0010392304845413265
            args.learning_rate_value = 0.0027213442056664362
            args.learning_rate_policy = 0.00024494897427831784 * 9
        exp = f"ppo-{'fixed1' if fixed_lr else 'adaptive'}-{nn_size}"
        run_evaluation(args=args, env=environment,
                       experiment=exp, steps=1_000_000,
                       parallel_runs=6, runs=1)


def nn_size_comp_with_horizon():
    environment = "Hopper-v2"
    for nn_size in [1, 5, 10, 15, 20, 25, 30]:
        args = get_default_args()
        args.horizon = 4096
        args.nn_size = nn_size
        run_evaluation(args=args, env=environment,
                       experiment=f"ppo-adaptive-{nn_size}-horizon", steps=1_000_000, parallel_runs=6)


def nn_size_comp_without_dynamic_step():
    args = get_default_args()
    args.nn_size = 29
    run_evaluation(args=args, env="Hopper-v2",
                   experiment=f"ppo-adaptive-{29}-nds", steps=1_000_000, parallel_runs=3, runs=2)


def environments(environments):
    for environment in environments:
        args = get_default_args()
        run_evaluation(args=args, env=environment,
                       experiment=f"ppo-2048-horizon", steps=1_000_000, parallel_runs=4)


def horizons(horizons):
    for horizon in horizons:
        args = get_default_args()
        args.horizon = horizon
        run_evaluation(args=args, env="Hopper-v2",
                       experiment=f"ppo-horizons-{horizon}", steps=1_000_000, parallel_runs=4)


def lr_value(lr_values):
    for lr_value in lr_values:
        args = get_default_args()
        args.lr_value = lr_value
        run_evaluation(args=args, env="Hopper-v2",
                       experiment=f"ppo-lr-value-{lr_value}", steps=1_000_000, parallel_runs=4)


def environments(environments):
    for environment in environments:
        args = get_default_args()
        run_evaluation(args=args, env=environment,
                       experiment=f"ppo-test-adaptive-clipping2", steps=1_000_000, parallel_runs=6)


def test_loss(environments):
    for environment in environments:
        args = get_default_args()
        for loss in ["adaptive_kl", "clipped_kl", "surrogate", "pg"]:
            args.policy_loss = loss
            run_evaluation(args=args, env=environment,
                           experiment=f"ppo-test-loss-{loss}", steps=1_000_000, parallel_runs=6)


def test_relu(environments):
    for environment in environments:
        args = get_default_args()
        args.learning_rate_value = 0.0014696938456699071
        args.learning_rate_policy = 7.745966692414833e-05
        args.activation_function = "relu"
        run_evaluation(args=args, env=environment,
                       experiment=f"ppo-test-relu", steps=1_000_000, parallel_runs=6)


def test_entropy_coef(environments):
    for environment in environments:
        for coef in [0.1, 0.25, 0.5, 0.75, 1.0]:
            args = get_default_args()
            args.policy_entropy_coef = coef
            run_evaluation(args=args, env=environment,
                           experiment=f"ppo-test-entropy_coef{coef}", steps=1_000_000, parallel_runs=6)


def test_value_coef(environments):
    for environment in environments:
        args = get_default_args()
        args.vf_coef = 0.5
        run_evaluation(args=args, env=environment,
                       experiment=f"ppo-test-value_coef", steps=1_000_000, parallel_runs=6)


def test_kernel_initializer(environments):
    for environment in environments:
        for initializer in [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]:
            args = get_default_args()
            args.kernel_initializer = initializer
            run_evaluation(args=args, env=environment,
                           experiment=f"ppo-test-kernel-initializer-normal-{initializer}", steps=1_000_000,
                           parallel_runs=6)


def test_rollout_size(environments):
    for environment in environments:
        for s in [5, 10, 15]:
            args = get_default_args()
            args.rollout_size = s
            run_evaluation(args=args, env=environment,
                           experiment=f"ppo-test-rollout-size-{s}", steps=1_000_000, parallel_runs=6)


def test_horizon_cap_size(environments):
    for environment in environments:
        for s in [2048, 4096]:
            args = get_default_args()
            args.horizon = s
            args.rollout_size = 500  # set rollout size high
            run_evaluation(args=args, env=environment,
                           experiment=f"ppo-test-horizon-cap2-{s}", steps=1_000_000, parallel_runs=3)


def nn_deep_comp(fixed_lr=False, sizes=range(1, 30)):
    environment = "Hopper-v2"
    args = get_default_args()
    args.structure = "straight"
    for h, s in [(1, 64), (2, 64), (3, 64), (4, 64), (5, 64), (2, 32), (3, 16)]:
        args.hidden_layers = h
        args.straight_size = s
        run_evaluation(args=args, env=environment,
                       experiment=f"ppo-nn-deep-comp-h{args.hidden_layers}-s{args.straight_size}", steps=1_000_000,
                       parallel_runs=6, runs=1)

    args.hidden_layers = 5
    args.straight_size = 64
    args.activation_function = "relu"
    run_evaluation(args=args, env=environment,
                   experiment=f"ppo-nn-deep-comp-h{args.hidden_layers}-s{args.straight_size}", steps=1_000_000,
                   parallel_runs=6, runs=1)


default_environments = ["Hopper-v2", "Walker2d-v2", "FetchReach-v1", "InvertedDoublePendulum-v2"]

# uncomment experiment to run

test()
# environments(["Hopper-v2"])
# test_loss(default_environments)
# test_relu(default_environments)
# test_entropy_coef(default_environments)
# test_value_coef(default_environments)
# test_kernel_initializer(["Hopper-v2"])
# test_horizon_cap_size(["Hopper-v2"])
# nn_size_comp()
# test_kernel_initializer(default_environments)
# nn_size_comp_adaptive(fixed_lr=True, sizes=range(19, 30))
# nn_deep_comp()
# nn_size_comp(fixed_lr=True, sizes=[60, 90])
# nn_size_comp(fixed_lr=False, sizes=[60, 90])
# "FetchReach-v1", "InvertedDoublePendulum-v2", "Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"
# nn_size_comp_without_dynamic_step()
# nn_size_comp_with_horizon()
# environments(["Hopper-v2"])
# environments(["Hopper-v2"])
# horizons([2048, 4096, 8192, 16384])  # vf-test is 0.003
