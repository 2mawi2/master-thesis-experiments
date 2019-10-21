from collections import deque

try:
    import roboschool
except:
    pass
import numpy as np


def _run_episode(env, args, policy, preprocessor, render=False):
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = preprocessor.get()
    scale[-1] = 1.0
    offset[-1] = 0.0

    while not done:
        if render:
            env.render()
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
        actions.append(action)
        obs, reward, done, _ = env.step(action[0])

        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


reward_buffer = deque(maxlen=100)
length_buffer = deque(maxlen=100)


def generate_segment(env, args, policy, preprocessor, logger):
    total_steps = 0
    trajectories = []
    episodes = 0

    while total_steps <= args.horizon and episodes < args.rollout_size:
        observes, actions, rewards, unscaled_obs = _run_episode(env, args, policy, preprocessor,
                                                                render=args.render)
        trajectories.append({'observes': observes,
                             'actions': actions,
                             'rewards': rewards,
                             'unscaled_obs': unscaled_obs})

        total_steps += observes.shape[0]
        episodes += 1

    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    preprocessor.update(unscaled)  # update running statistics for scaling observations

    trajectory_mean_reward = np.mean([t['rewards'].sum() for t in trajectories])

    reward_buffer.append(trajectory_mean_reward)
    length_buffer.append(total_steps)

    logger.log({'MeanReward': trajectory_mean_reward,
                'steps': total_steps,
                'RollingMeanReward': np.mean(reward_buffer),
                'Rollingsteps': np.mean(length_buffer),
                })

    return trajectories, total_steps


def unpack_segment(segment):
    observes = np.concatenate([t['observes'] for t in segment])
    actions = np.concatenate([t['actions'] for t in segment])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in segment])
    advantages = np.concatenate([t['advantages'] for t in segment])
    return observes, actions, advantages, disc_sum_rew
