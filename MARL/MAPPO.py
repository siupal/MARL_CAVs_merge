import torch as th
from torch import nn
import configparser

config_dir = 'configs/configs_ppo.ini'                                              # 配置文件的路径
config = configparser.ConfigParser()                                                # 创建配置文件解析器
config.read(config_dir)                                                             # 读取配置文件
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')                            # 获取model_config部分的torch_seed的值，并返回为整型
th.manual_seed(torch_seed)                                                          # 以torch_seed的值为种子，初始化生成随机数生成器的种子
th.backends.cudnn.benchmark = False                                                 # 为cudnn的后端设置benchmark属性为False
th.backends.cudnn.deterministic = True                                              # 为cudnn的后端设置deterministic属性为True

from torch.optim import Adam, RMSprop                                               # 从torch.optim中导入Adam和RMSprop优化器

import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Memory_common import OnPolicyReplayMemory
from single_agent.Model_common import ActorNetwork, CriticNetwork
from common.utils import index_to_one_hot, to_tensor_var, VideoRecorder


class MAPPO:
    """
    An multi-agent learned with PPO
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="global_R"):                             # 初始化MAPPO类

        assert traffic_density in [1, 2, 3]                                                             # 断言traffic_density的值为1,2,3
        assert reward_type in ["regionalR", "global_R"]                                                 # 断言reward_type的值为"regionalR"或"global_R"
        self.reward_type = reward_type
        self.env = env
        self.state_dim = state_dim                                                                      # 状态维度
        self.action_dim = action_dim                                                                    # 动作维度
        self.env_state, self.action_mask = self.env.reset()                                             # 初始化环境状态和动作掩码
        self.n_episodes = 0                                                                             # 记录智能体与环境交互的轮数
        self.n_steps = 0                                                                                # 记录智能体与环境交互的步数
        self.max_steps = max_steps                                                                      # 最大步数
        self.test_seeds = test_seeds                                                                    # 测试种子
        self.reward_gamma = reward_gamma                                                                # 奖励折扣
        self.reward_scale = reward_scale                                                                # 奖励缩放
        self.traffic_density = traffic_density                                                          # 记录交通密度
        self.memory = OnPolicyReplayMemory(memory_capacity)                                             # 创建OnPolicyReplayMemory对象
        self.actor_hidden_size = actor_hidden_size                                                      # actor隐藏层大小
        self.critic_hidden_size = critic_hidden_size                                                    # critic隐藏层大小
        self.actor_output_act = actor_output_act                                                        # actor输出激活函数
        self.critic_loss = critic_loss                                                                  # critic损失函数
        self.actor_lr = actor_lr                                                                        # actor学习率
        self.critic_lr = critic_lr                                                                      # critic学习率
        self.optimizer_type = optimizer_type                                                            # 优化器类型
        self.entropy_reg = entropy_reg                                                                  # 熵正则化
        self.max_grad_norm = max_grad_norm                                                              # 最大梯度范数
        self.batch_size = batch_size                                                                    # 批大小
        self.episodes_before_train = episodes_before_train                                              # 训练之前的轮数
        self.use_cuda = use_cuda and th.cuda.is_available()                                             # 判断是否使用cuda
        self.roll_out_n_steps = roll_out_n_steps                                                        # roll_out_n_steps
        self.target_tau = target_tau                                                                    # 目标tau
        self.target_update_steps = target_update_steps                                                  # 目标更新步数
        self.clip_param = clip_param                                                                    # PPO的clip参数

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)                               # 创建actor网络
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)        # 创建critic网络
        # to ensure target network and learning network has the same weights
        # 共享参数的方法来保证目标网络和学习网络有相同的权重
        self.actor_target = deepcopy(self.actor)                                                        # 深拷贝actor网络
        self.critic_target = deepcopy(self.critic)                                                      # 深拷贝critic网络

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)                      # 创建actor优化器，Adam学习率为actor的学习率actor_lr
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)                   # 创建critic优化器，Adam学习率为critic的学习率critic_lr
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)                   # 创建actor优化器，RMSprop学习率为actor的学习率actor_lr
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)                # 创建critic优化器，RMSprop学习率为critic的学习率critic_lr

        # 说明要么使用Adam优化器，要么使用RMSprop优化器

        if self.use_cuda:   # use GPU
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        # 若有GPU，则将actor、critic、actor_target、critic_target网络放到GPU上运行以加速运算

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]

    # agent interact with the environment to collect experience 智能体与环境交互以收集经验
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):                   # 判断智能体与环境交互的步数是否达到最大步数
            self.env_state, _ = self.env.reset()                                                # 初始化/重置环境
            self.n_steps = 0                                                                    # 初始化/重置智能体与环境交互的步数
        states = []
        actions = []
        rewards = []                                                                            # 创建空列表用于存储智能体的状态、动作和奖励
        done = True                                                                             # 记录智能体与环境交互是否结束
        average_speed = 0                                                                       # 重置平均速度

        self.n_agents = len(self.env.controlled_vehicles)                                       # 获取环境中的智能体数量

        # take n steps

        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)                                                       # 将环境状态添加到states列表中
            action = self.exploration_action(self.env_state, self.n_agents)                     # 根据环境状态和智能体数量获取动作
            next_state, global_reward, done, info = self.env.step(tuple(action))                # 获取下一个状态、全局奖励、done标志和信息（以动作元组的长度作为步长？）
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])              # 将动作添加到actions列表中
            self.episode_rewards[-1] += global_reward                                           # 计算全局奖励
            self.epoch_steps[-1] += 1                                                           # 计算步数
            if self.reward_type == "regionalR":                                                 # 如果奖励类型为"regionalR"
                reward = info["regional_rewards"]                                               # 令区域奖励值的信息为奖励
            elif self.reward_type == "global_R":                                                # 如果奖励类型为"global_R"
                reward = [global_reward] * self.n_agents                                        # 则计算全局奖励
            rewards.append(reward)                                                              # 将奖励添加到rewards列表中
            average_speed += info["average_speed"]                                              # 计算平均速度
            final_state = next_state                                                            # 记录最终状态
            self.env_state = next_state                                                         # 更新环境状态

            # print(self.env_state)  type() .shape len() a[:1,:,:]
            #self.env.render()

            self.n_steps += 1                                                                   # 计算智能体与环境交互的步数计数
            if done:
                self.env_state, _ = self.env.reset()                                            # 重置环境
                break

        # discount reward   折扣奖励
        if done:
            final_value = [0.0] * self.n_agents                                                 # 创建长度为n_agents的列表，用于存储最终价值
            self.n_episodes += 1                                                                # 计算智能体与环境交互的轮数
            self.episode_done = True                                                            # 记录智能体与环境交互是否结束
            self.episode_rewards.append(0)                                                      # 重置轮数奖励
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]                       # 计算平均速度
            self.average_speed.append(0)                                                        # 重置平均速度
            self.epoch_steps.append(0)                                                          # 重置步数
        else:
            self.episode_done = False                                                           # 记录智能体与环境交互是否结束
            final_action = self.action(final_state)                                             # 根据最终状态获取最终动作
            final_value = self.value(final_state, final_action)                                 # 根据最终状态和最终动作获取最终价值

        if self.reward_scale > 0:                                                               # 如果奖励缩放大于0
            rewards = np.array(rewards) / self.reward_scale                                     # 则将奖励缩放

        for agent_id in range(self.n_agents):                                                               # 遍历智能体
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_value[agent_id])       # 折扣奖励

        rewards = rewards.tolist()                                                              # 将rewards转换为列表
        self.memory.push(states, actions, rewards)                                              # 将states、actions和rewards添加到memory中

    # train on a roll out batch
    def train(self):                                                                            # 训练
        if self.n_episodes <= self.episodes_before_train:                                       # 如果智能体与环境交互的轮数小于等于训练之前的轮数
            pass                                                                                # 则不进行训练

        batch = self.memory.sample(self.batch_size)                                             # 从memory中采样batch_size大小的样本
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)             # 将states转换为张量
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)          # 将actions转换为张量
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)                        # 将rewards转换为张量

        for agent_id in range(self.n_agents):                                                                       # 遍历智能体
            # update actor network
            self.actor_optimizer.zero_grad()                                                                        # 清空actor优化器的梯度
            values = self.critic_target(states_var[:, agent_id, :], actions_var[:, agent_id, :]).detach()           # 获取目标网络的价值
            advantages = rewards_var[:, agent_id, :] - values                                                       # 计算优势

            action_log_probs = self.actor(states_var[:, agent_id, :])                                               # 获取动作的对数概率
            action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)                            # 计算动作的对数概率
            old_action_log_probs = self.actor_target(states_var[:, agent_id, :]).detach()                           # 获取目标网络的动作对数概率
            old_action_log_probs = th.sum(old_action_log_probs * actions_var[:, agent_id, :], 1)                    # 计算目标网络的动作对数概率
            ratio = th.exp(action_log_probs - old_action_log_probs)                                                 # 计算比率
            surr1 = ratio * advantages                                                                              # PPO's optimality surrogate (L^CLIP)
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages                      # PPO截断
            # PPO's pessimistic surrogate (L^CLIP)
            actor_loss = -th.mean(th.min(surr1, surr2))                                                             # 计算actor损失函数值
            actor_loss.backward()                                                                                   # 反向传播
            if self.max_grad_norm is not None:                                                                      # 如果最大梯度范数不为空
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)                               # 则对梯度进行裁剪
            self.actor_optimizer.step()                                                                             # 更新actor网络

            # update critic network
            self.critic_optimizer.zero_grad()                                                                       # 清空critic优化器的梯度
            target_values = rewards_var[:, agent_id, :]                                                             # 目标值
            values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])                           # 计算价值
            if self.critic_loss == "huber":                                                                         # 如果critic损失函数为huber
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)                                   # 计算critic损失
            else:
                critic_loss = nn.MSELoss()(values, target_values)                                                   # 计算critic损失
            critic_loss.backward()                                                                                  # 反向传播
            if self.max_grad_norm is not None:                                                                      # 如果最大梯度范数不为空
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)                              # 则对梯度进行裁剪
            self.critic_optimizer.step()                                                                            # 更新critic网络

        # update actor target network and critic target network
        if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:                                 # 如果智能体与环境交互的轮数能够被目标更新步数整除且大于0
            self._soft_update_target(self.actor_target, self.actor)                                                 # 软更新目标网络
            self._soft_update_target(self.critic_target, self.critic)                                               # 软更新目标网络

    # predict softmax action based on state
    def _softmax_action(self, state, n_agents):                                                 # 获取softmax动作
        state_var = to_tensor_var([state], self.use_cuda)                                       # 将state转换为张量

        softmax_action = []                                                                     # 创建空列表用于存储softmax动作
        for agent_id in range(n_agents):                                                        # 遍历智能体
            softmax_action_var = th.exp(self.actor(state_var[:, agent_id, :]))                  # 计算softmax动作

            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])                 # 将softmax动作添加到softmax_action列表中
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])                       # 将softmax动作添加到softmax_action列表中
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, n_agents):                                              # 选择基于状态的动作，在训练中添加随机噪声以进行探索
        softmax_actions = self._softmax_action(state, n_agents)                                 # 获取softmax动作
        actions = []                                                                            # 创建空列表用于存储动作
        for pi in softmax_actions:                                                              # 遍历softmax动作
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))                          # 选择动作
        return actions                                                                          # 返回动作

    # choose an action based on state for execution
    def action(self, state, n_agents):                                                          # 选择基于状态的动作以执行
        softmax_actions = self._softmax_action(state, n_agents)                                 # 获取softmax动作
        actions = []                                                                            # 创建空列表用于存储动作
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)                                       # 将state转换为张量
        action = index_to_one_hot(action, self.action_dim)                                      # 将动作转换为one-hot编码
        action_var = to_tensor_var([action], self.use_cuda)                                     # 将动作转换为张量

        values = [0] * self.n_agents                                                            # 创建长度为n_agents的列表，用于存储价值
        for agent_id in range(self.n_agents):                                                   # 遍历智能体
            value_var = self.critic(state_var[:, agent_id, :], action_var[:, agent_id, :])      # 计算价值

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]                              # 将价值添加到values列表中
            else:
                values[agent_id] = value_var.data.numpy()[0]                                    # 将价值添加到values列表中
        return values                                                                           # 返回价值

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):                      # 对智能体的评估环节
        rewards = []
        infos = []                                                                              # 创建空列表用于存储信息
        avg_speeds = []                                                                         # 信息
        steps = []                                                                              # 步数
        vehicle_speed = []                                                                      # 车速
        vehicle_position = []                                                                   # 位置
        video_recorder = None                                                                   # 初始化视频记录器
        seeds = [int(s) for s in self.test_seeds.split(',')]                                    # 将测试种子切片后转换为整型

        for i in range(eval_episodes):                                                                              # 遍历评估轮数
            avg_speed = 0                                                                                           # 初始化平均速度
            step = 0                                                                                                # 初始化步数
            rewards_i = []                                                                                          # 创建空列表用于存储奖励
            infos_i = []                                                                                            # 创建空列表用于存储信息
            done = False                                                                                            # 初始化done标志
            if is_train:                                                                                            # 不同环境下的训练
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 4)

                # 对训练环境进行重建

            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])                           # 对测试环境进行重建

            n_agents = len(env.controlled_vehicles)                                                                 # 获取环境中的智能体数量
            rendered_frame = env.render(mode="rgb_array")                                                           # 每一轮评估时都进行渲染
            video_filename = os.path.join(output_dir,
                                          "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                          '.mp4')                                                                   # 保存训练视频
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                      5))                                           # 输出视频信息
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=5)                              # 创建视频记录器
                video_recorder.add_frame(rendered_frame)                                                            # 添加帧
            else:
                video_recorder = None                                                                               # 否则视频记录器为空

            while not done:                                                                                         # 当done为False时
                step += 1                                                                                           # 计算步数
                action = self.action(state, n_agents)                                                               # 获取动作
                state, reward, done, info = env.step(action)                                                        # 获取下一个状态、奖励、done标志和信息
                avg_speed += info["average_speed"]                                                                  # 计算平均速度
                rendered_frame = env.render(mode="rgb_array")                                                       # 每一轮评估时都进行渲染
                if video_recorder is not None:                                                                      # 如果视频记录器不为空
                    video_recorder.add_frame(rendered_frame)                                                        # 根据渲染添加帧

                rewards_i.append(reward)                                                                            # 将奖励添加到rewards_i列表中
                infos_i.append(info)                                                                                # 将信息添加到infos_i列表中

            vehicle_speed.append(info["vehicle_speed"])                                                             # 将车速添加到vehicle_speed列表中
            vehicle_position.append(info["vehicle_position"])                                                       # 将位置添加到vehicle_position列表中
            rewards.append(rewards_i)                                                                               # 将rewards_i添加到rewards列表中
            infos.append(infos_i)                                                                                   # 将infos_i添加到infos列表中
            steps.append(step)                                                                                      # 将步数添加到steps列表中
            avg_speeds.append(avg_speed / step)                                                                     # 计算平均速度

        if video_recorder is not None:                                                                              # 如果视频记录器不为空
            video_recorder.release()                                                                                # 释放视频记录器
        env.close()                                                                                                 # 关闭环境
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds                                        # 返回奖励、车速、位置、步数和平均速度

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):                                                               # 折扣奖励
        discounted_r = np.zeros_like(rewards)                                                                       # 创建与rewards相同大小的零数组
        running_add = final_value                                                                                   # 计算最终价值
        for t in reversed(range(0, len(rewards))):                                                                  # 反向遍历rewards
            running_add = running_add * self.reward_gamma + rewards[t]                                              # 计算折扣奖励
            discounted_r[t] = running_add                                                                           # 计算折扣奖励
        return discounted_r                                                                                         # 返回折扣奖励

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):                                                                  # 软更新目标网络
        for t, s in zip(target.parameters(), source.parameters()):                                                  # 遍历目标网络和源网络的参数
            t.data.copy_(                                                                                           # 将源网络的参数复制到目标网络的参数
                (1. - self.target_tau) * t.data + self.target_tau * s.data)                                         # 计算目标网络的参数

    def load(self, model_dir, global_step=None, train_mode=False):                                                  # 加载模型
        save_file = None                                                                                            # 初始化保存文件
        save_step = 0                                                                                               # 初始化保存步数
        if os.path.exists(model_dir):                                                                               # 如果模型目录存在
            if global_step is None:                                                                                 # 如果全局步数为空
                for file in os.listdir(model_dir):                                                                  # 遍历模型目录
                    if file.startswith('checkpoint'):                                                               # 如果文件名以checkpoint开头
                        tokens = file.split('.')[0].split('-')                                                      # 切片
                        if len(tokens) != 2:                                                                        # 如果长度不等于2
                            continue                                                                                # 继续遍历模型目录
                        cur_step = int(tokens[1])                                                                   # 获取当前步数
                        if cur_step > save_step:                                                                    # 如果当前步数大于保存步数
                            save_file = file                                                                        # 保存文件
                            save_step = cur_step                                                                    # 保存步数
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)                                                # 保存文件
        if save_file is not None:                                                                                   # 如果保存文件不为空
            file_path = model_dir + save_file                                                                       # 文件路径
            checkpoint = th.load(file_path)                                                                         # 加载文件
            print('Checkpoint loaded: {}'.format(file_path))                                                        # 输出加载文件信息
            self.actor.load_state_dict(checkpoint['model_state_dict'])                                              # 加载actor网络的参数
            if train_mode:                                                                                          # 如果是训练模式
                self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                            # 加载actor优化器的参数
                self.actor.train()                                                                                  # 训练
            else:
                self.actor.eval()                                                                                   # 否则评估（要么训练，要么评估）
            return True                                                                                             # 返回True
        logging.error('Can not find checkpoint for {}'.format(model_dir))                                           # 输出错误信息
        return False                                                                                                # 返回False

    def save(self, model_dir, global_step):                                                                         # 保存模型
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)                                            # 文件路径
        th.save({'global_step': global_step,                                                                        # 保存全局步数
                 'model_state_dict': self.actor.state_dict(),                                                       # 保存actor网络的参数
                 'optimizer_state_dict': self.actor_optimizer.state_dict()},                                        # 保存actor优化器的参数
                file_path)                                                                                          # 保存文件
