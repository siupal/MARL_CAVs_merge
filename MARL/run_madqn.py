from MADQN import MADQN
from single_agent.utils_common import agg_double_list

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'highway-env'))

# 添加警告过滤，隐藏所有警告信息
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
from tqdm import tqdm

# 添加numpy兼容层，解决bool8属性错误
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import matplotlib.pyplot as plt
import highway_env
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MAX_EPISODES = 20000
EPISODES_BEFORE_TRAIN = 10
EVAL_EPISODES = 3
EVAL_INTERVAL = 200

# max steps in each episode, prevent from running too long
MAX_STEPS = 100

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 128
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 50000


def run():
    env = gym.make('merge-multi-agent-v0')
    env_eval = gym.make('merge-multi-agent-v0')
    state_dim = env.n_s
    action_dim = env.n_a

    madqn = MADQN(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_dim, action_dim=action_dim,
              batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN)

    episodes = []
    eval_rewards = []
    
    # 使用tqdm创建进度条
    progress_bar = tqdm(total=MAX_EPISODES, desc="训练进度", unit="episode")
    progress_bar.update(madqn.n_episodes)  # 更新当前进度
    
    while madqn.n_episodes < MAX_EPISODES:
        madqn.interact()
        if madqn.n_episodes >= EPISODES_BEFORE_TRAIN:
            madqn.train()
        if madqn.episode_done:
            # 更新进度条
            progress_bar.update(1)
            if hasattr(madqn, 'episode_rewards') and len(madqn.episode_rewards) > 0:
                progress_bar.set_postfix({"reward": madqn.episode_rewards[-1]})
            
            if ((madqn.n_episodes + 1) % EVAL_INTERVAL == 0):
                rewards, _ = madqn.evaluation(env_eval, EVAL_EPISODES)
                rewards_mu, rewards_std = agg_double_list(rewards)
                print("\nEpisode %d, Average Reward %.2f" % (madqn.n_episodes + 1, rewards_mu))
                episodes.append(madqn.n_episodes + 1)
                eval_rewards.append(rewards_mu)
                # 更新进度条描述
                progress_bar.set_description(f"训练进度 [平均奖励: {rewards_mu:.2f}]")

    # 关闭进度条
    progress_bar.close()
    
    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["DQN"])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()
