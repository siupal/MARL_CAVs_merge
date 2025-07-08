from MAPPO import MAPPO         # from common.mappo import MAPPO
from common.utils import agg_double_list, copy_file_ppo, init_dir
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'highway-env'))

import gym
import numpy as np

# 添加numpy兼容层，解决bool8属性错误
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import matplotlib.pyplot as plt
import highway_env
import argparse
import configparser
import os
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_args():                                                                                   # 解析参数
    """
    Description for this experiment:
        + easy: globalR
        + seed = 0
    """
    default_base_dir = "./results/"                                                                 # 默认的保存路径
    default_config_dir = os.path.join(os.path.dirname(__file__), 'configs/configs_ppo.ini')  # 默认的配置文件路径
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using mappo'))                                   # 创建一个解析器，描述解析器的作用
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")                       # 添加参数
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")                                  # 添加参数
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")                  # 添加参数
    parser.add_argument('--model-dir', type=str, required=False,
                        default='', help="pretrained model path")                                   # 添加参数
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),                      # 添加参数
                        help="random seeds for evaluation, split by ,")
    args = parser.parse_args()                                                                      # 解析参数
    return args                                                                                     # 返回参数


def train(args):                                                                                    # 训练
    base_dir = args.base_dir                                                                        # 保存路径
    config_dir = args.config_dir                                                                    # 配置文件路径
    config = configparser.ConfigParser()                                                            # 创建一个配置解析器
    config.read(config_dir)                                                                         # 读取配置文件

    # create an experiment folder
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")                                              # 获取当前时间
    output_dir = base_dir + now                                                                     # 保存路径
    dirs = init_dir(output_dir)                                                                     # 初始化保存路径
    copy_file_ppo(dirs['configs'])                                                                  # 复制文件

    if os.path.exists(args.model_dir):                                                              # 如果模型路径存在
        model_dir = args.model_dir                                                                  # 模型路径
    else:                                                                                           # 否则
        model_dir = dirs['models']                                                                  # 模型路径

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')                                        # 批大小
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')                              # 记忆容量
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')                            # 滚动步数
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')                                  # 奖励gamma
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')                          # actor隐藏层大小
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')                        # critic隐藏层大小
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')                                # 最大梯度范数
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')                                    # 熵正则化
    reward_type = config.get('MODEL_CONFIG', 'reward_type')                                         # 奖励类型
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')                      # 目标更新步数
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')                                      # 目标tau

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')                                          # actor学习率
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')                                        # critic学习率
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')                                    # 最大episode
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')                  # 训练前的episode
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')                                  # 评估间隔
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')                                  # 评估episode
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')                                  # 奖励缩放

    # init env
    env = gym.make('merge-multi-agent-v0')                                                          # 创建环境
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')                                        # 种子
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')        # 仿真频率
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')                                # 持续时间
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')                # 策略频率
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')                # 碰撞奖励
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')              # 高速奖励
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')                        # 跟车成本
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')                      # 跟车时间
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')              # 合并车道成本
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')                  # 交通密度
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')                                # 交通密度
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')              # 动作屏蔽

    assert env.T % ROLL_OUT_N_STEPS == 0                                                            # 断言        # 确保T能被滚动步数整除

    #评估环境
    env_eval = gym.make('merge-multi-agent-v0')                                                     # 创建环境
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1                               # 种子
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')   # 仿真频率
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')                           # 持续时间
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')           # 策略频率
    env_eval.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')           # 碰撞奖励
    env_eval.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')         # 高速奖励
    env_eval.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')                   # 跟车成本
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')                 # 跟车时间
    env_eval.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')         # 合并车道成本
    env_eval.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')             # 交通密度
    env_eval.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')         # 动作屏蔽

    state_dim = env.n_s                                                                             # 状态维度
    action_dim = env.n_a                                                                            # 动作维度
    test_seeds = args.evaluation_seeds                                                              # 测试种子

    mappo = MAPPO(env=env, memory_capacity=MEMORY_CAPACITY,
                  state_dim=state_dim, action_dim=action_dim,
                  batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                  target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                  reward_gamma=reward_gamma, reward_type=reward_type,
                  max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                  episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density
                  )                                                                                 # 创建MAPPO对象 # 初始化

    # load the model if exist
    mappo.load(model_dir, train_mode=True)                                                          # 加载模型  # 训练模式  # 如果存在  # 模型路径
    env.seed = env.config['seed']                                                                   # 设置种子
    env.unwrapped.seed = env.config['seed']                                                         # 设置种子
    eval_rewards = []                                                                               # 评估奖励  # 空列表，用于存放评估奖励

    while mappo.n_episodes < MAX_EPISODES:                                                          # 当episode小于最大episode时
        mappo.interact()                                                                            # 交互
        if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:                                               # 如果episode大于等于训练前的episode
            mappo.train()                                                                           # 训练
        if mappo.episode_done and ((mappo.n_episodes + 1) % EVAL_INTERVAL == 0):                    # 如果episode结束  # 并且  # 模型的episode+1能被评估间隔整除
            rewards, _, _, _ = mappo.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)      # 评估  # 评估环境  # 保存路径  # 评估episode
            rewards_mu, rewards_std = agg_double_list(rewards)                                      # 奖励均值  # 奖励标准差
            print("Episode %d, Average Reward %.2f" % (mappo.n_episodes + 1, rewards_mu))           # 打印
            eval_rewards.append(rewards_mu)                                                         # 评估奖励列表添加奖励均值
            # save the model
            mappo.save(dirs['models'], mappo.n_episodes + 1)                                        # 保存模型  # 保存路径  # episode+1

    # save the model
    mappo.save(dirs['models'], MAX_EPISODES + 2)                                                    # 保存模型  # 保存路径  # 最大episode+2

    plt.figure()                                                                                    # 创建图
    plt.plot(eval_rewards)                                                                          # 画图
    plt.xlabel("Episode")                                                                           # x轴
    plt.ylabel("Average Reward")                                                                    # y轴
    plt.legend(["MAPPO"])                                                                           # 图例
    plt.show()                                                                                      # 显示图


def evaluate(args):                                                                                 # 评估
    if os.path.exists(args.model_dir):                                                              # 如果模型路径存在
        model_dir = args.model_dir + '/models/'                                                     # 模型路径
    else:                                                                                           # 否则
        raise Exception("Sorry, no pretrained models")                                              # 抛出异常
    config_dir = args.model_dir + '/configs/configs_ppo.ini'                                        # 配置文件路径
    config = configparser.ConfigParser()                                                            # 创建一个配置解析器
    config.read(config_dir)                                                                         # 读取配置文件

    video_dir = args.model_dir + '/eval_videos'                                                     # 视频路径

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')                                        # 批大小
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')                              # 记忆容量
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')                            # 滚动步数
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')                                  # 奖励gamma
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')                          # actor隐藏层大小
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')                        # critic隐藏层大小
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')                                # 最大梯度范数
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')                                    # 熵正则化
    reward_type = config.get('MODEL_CONFIG', 'reward_type')                                         # 奖励类型
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')                      # 目标更新步数
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')                                      # 目标tau

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')                                          # actor学习率
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')                                        # critic学习率
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')                  # 训练前的episode
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')                                  # 奖励缩放

    # init env
    env = gym.make('merge-multi-agent-v0')                                                          # 创建环境
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')                                        # 种子
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')        # 仿真频率
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')                                # 持续时间
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')                # 策略频率
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')                # 碰撞奖励
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')              # 高速奖励
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')                        # 跟车成本
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')                      # 跟车时间
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')              # 合并车道成本
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')                  # 交通密度
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')                                # 交通密度
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')              # 动作屏蔽

    assert env.T % ROLL_OUT_N_STEPS == 0                                                            # 断言
    state_dim = env.n_s                                                                             # 状态维度
    action_dim = env.n_a                                                                            # 动作维度
    test_seeds = args.evaluation_seeds                                                              # 测试种子
    seeds = [int(s) for s in test_seeds.split(',')]                                                 # 种子列表

    mappo = MAPPO(env=env, memory_capacity=MEMORY_CAPACITY,
                  state_dim=state_dim, action_dim=action_dim,
                  batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                  target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                  reward_gamma=reward_gamma, reward_type=reward_type,
                  max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                  episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density
                  )                                                                                 # 创建MAPPO对象

    # load the model if exist
    mappo.load(model_dir, train_mode=False)                                                         # 加载模型  # 训练模式  # 如果存在  # 模型路径
    rewards, _, steps, avg_speeds = mappo.evaluation(env, video_dir, len(seeds), is_train=False)    # 评估  # 评估环境  # 视频路径  # 种子长度  # 训练模式


if __name__ == "__main__":                                                                          # 如果是主程序
    args = parse_args()                                                                             # 解析参数
    # train or eval
    if args.option == 'train':                                                                      # 如果是训练
        train(args)                                                                                 # 训练
    else:
        evaluate(args)                                                                              # 评估
