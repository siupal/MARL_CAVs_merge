import cv2, os
import torch as th
from torch.autograd import Variable
import numpy as np
from shutil import copy
import torch.nn as nn


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1) * (log_p2 - log_p1), 1)


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def index_to_one_hot(index, dim):
    # if isinstance(index, np.int) or isinstance(index, np.int64):
    #     one_hot = np.zeros(dim)
    #     one_hot[index] = 1.
    # else:
    #     one_hot = np.zeros((len(index), dim))
    #     one_hot[np.arange(len(index)), index] = 1.
    # return one_hot

    ss = str(index)
    one_hot = np.zeros((len(ss), dim))
    one_hot[np.arange(len(ss)), index] = 1.
    return one_hot


def to_tensor_var(x, use_cuda=True, dtype="float"):                                         # 将数据转换为张量
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor                       # 如果使用cuda，则使用cuda.FloatTensor，否则使用FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor                          # 如果使用cuda，则使用cuda.LongTensor，否则使用LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor                          # 如果使用cuda，则使用cuda.ByteTensor，否则使用ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()                                          # 将x转换为np.float64类型，并转换为列表
        return Variable(FloatTensor(x))                                                     # 返回x的张量
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()                                             # 将x转换为np.long类型，并转换为列表
        return Variable(LongTensor(x))                                                      # 返回x的张量
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()                                             # 将x转换为np.byte类型，并转换为列表
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()                                          # 将x转换为np.float64类型，并转换为列表
        return Variable(FloatTensor(x))


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std


class VideoRecorder:
    """This is used to record videos of evaluations"""

    def __init__(self, filename, frame_size, fps):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"MPEG"), int(fps),
            (frame_size[1], frame_size[0]))

    def add_frame(self, frame):                                                                         # 添加帧
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()                                                                     # 释放视频写入器

    def __del__(self):
        self.release()


def copy_file(tar_dir):
    # env = '.highway-env/envs/common/abstract.py'
    # copy(env, tar_dir)
    # env1 = '.highway_env/envs/merge_env_v1.py'
    # copy(env1, tar_dir)

    env2 = 'configs/configs.ini'
    copy(env2, tar_dir)

    models = 'MAA2C.py'
    copy(models, tar_dir)
    main = 'run_ma2c.py'
    copy(main, tar_dir)
    c1 = 'common/Agent.py'
    copy(c1, tar_dir)
    c2 = 'common/Memory.py'
    copy(c2, tar_dir)
    c3 = 'common/Model.py'
    copy(c3, tar_dir)


def copy_file_ppo(tar_dir):
    # env = '.highway-env/envs/common/abstract.py'
    # copy(env, tar_dir)
    # env1 = '.highway_env/envs/merge_env_v1.py'
    # copy(env1, tar_dir)

    env2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs/configs_ppo.ini')
    copy(env2, tar_dir)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    models = os.path.join(base_dir, 'MAPPO.py')
    copy(models, tar_dir)
    main = os.path.join(base_dir, 'run_mappo.py')
    copy(main, tar_dir)
    c1 = os.path.join(base_dir, 'single_agent/Agent_common.py')
    copy(c1, tar_dir)
    c2 = os.path.join(base_dir, 'single_agent/Memory_common.py')
    copy(c2, tar_dir)
    c3 = os.path.join(base_dir, 'single_agent/Model_common.py')
    copy(c3, tar_dir)


def copy_file_akctr(tar_dir):
    # env = '.highway-env/envs/common/abstract.py'
    # copy(env, tar_dir)
    # env1 = '.highway_env/envs/merge_env_v1.py'
    # copy(env1, tar_dir)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    env2 = os.path.join(base_dir, 'configs/configs_acktr.ini')
    copy(env2, tar_dir)

    models = os.path.join(base_dir, 'MAACKTR.py')
    copy(models, tar_dir)
    main = os.path.join(base_dir, 'run_maacktr.py')
    copy(main, tar_dir)
    c1 = os.path.join(base_dir, 'single_agent/Agent_common.py')
    copy(c1, tar_dir)
    c2 = os.path.join(base_dir, 'single_agent/Memory_common.py')
    copy(c2, tar_dir)
    c3 = os.path.join(base_dir, 'single_agent/Model_common.py')
    copy(c3, tar_dir)


def init_dir(base_dir, pathes=['train_videos', 'configs', 'models', 'eval_videos', 'eval_logs']):
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs
