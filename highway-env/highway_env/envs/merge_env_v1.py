"""
This environment is built on HighwayEnv with one main road and one merging lane.
Dong Chen: chendon9@msu.edu
Date: 01/05/2021
"""
import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:                                                                            # 默认配置
        config = super().default_config()                                                                       # 调用父类的默认配置
        config.update({                                                                                         # 更新配置
            "observation": {
                "type": "Kinematics"},
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,
            "screen_width": 600,
            "screen_height": 120,
            "centering_position": [0.3, 0.5],
            "scaling": 3,
            "simulation_frequency": 15,  # [Hz]
            "duration": 20,  # time step
            "policy_frequency": 5,  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 200,  # default=200
            "HIGH_SPEED_REWARD": 1,  # default=0.5
            "HEADWAY_COST": 4,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 4,  # default=4
            "traffic_density": 1,  # easy or hard modes
        })
        return config                                                                                           # 返回配置

    def _reward(self, action: int) -> float:                                                                    # 计算奖励
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)                                                                  # 返回奖励

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        # the optimal reward is 0
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])                     # 速度归一化
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 1):                                                                 # 车辆在合流道上
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))                                                                         # 合流道上的车辆
        else:
            Merging_lane_cost = 0                                                                               # 主干道上的车辆

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)                                              # 计算车辆的车头距离
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0       # 计算车头距离的奖励
        # compute overall reward
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)                      # 计算总的奖励
        return reward                                                                                           # 返回奖励

    def _regional_reward(self):                                                                                 # 区域奖励
        for vehicle in self.controlled_vehicles:                                                                # 对于每一个车辆
            neighbor_vehicle = []                                                                               # 邻近车辆

            # vehicle is on the main road
            if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
                    "c", "d", 0):                                                                               # 车辆在主干道上
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)                                            # 前车和后车
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:                                  # 如果车道不是最左边的车道
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])                     # 右边的车道
                # assume we can observe the ramp on this road
                elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > self.ends[0]:                # 如果车辆在合流道上
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))                         # 右边的车道
                else:
                    v_fr, v_rr = None, None                                                                     # 否则没有右边的车道
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)                                            # 前车和后车
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:                                  # 如果车道不是最左边的车道
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])                     # 左边的车道
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):                                                       # 如果车辆在主干道上
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))                         # 左边的车道
                else:
                    v_fl, v_rl = None, None                                                                     # 否则没有左边的车道
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:                                                         # 对于每一个车辆
                if type(v) is MDPVehicle and v is not None:                                                     # 如果车辆是MDPVehicle类型
                    neighbor_vehicle.append(v)                                                                  # 添加车辆
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)                                     # 区域奖励
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:                                        # 步骤
        agent_info = []                                                                                         # 代理信息
        obs, reward, done, info = super().step(action)                                                          # 调用父类的步骤
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)  # 代理是否终止
        for v in self.controlled_vehicles:                                                                      # 对于每一个车辆
            agent_info.append([v.position[0], v.position[1], v.speed])                                          # 添加车辆的位置和速度
        info["agents_info"] = agent_info                                                                        # 智能体信息

        for vehicle in self.controlled_vehicles:                                                                # 对于每一个车辆
            vehicle.local_reward = self._agent_reward(action, vehicle)                                          # 计算局部奖励
        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)            # 智能体奖励
        # regional reward
        self._regional_reward()                                                                                 # 区域奖励
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)       # 区域奖励

        obs = np.asarray(obs).reshape((len(obs), -1))                                                           # 观测
        return obs, reward, done, info                                                                          # 返回观测，奖励，是否结束，信息

    def _is_terminal(self) -> bool:                                                                             # 是否终止
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]                       # 或者超过时间

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:                                                     # 智能体是否终止
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]                       # 或者超过时间

    def _reset(self, num_CAV=0) -> None:                                                                        # 重置
        self._make_road()                                                                                       # 创建道路

        if self.config["traffic_density"] == 1:                                                                 # 交通密度
            # easy mode: 1-3 CAVs + 1-3 HDVs
            if num_CAV == 0:                                                                                    # 如果没有CAV
                num_CAV = np.random.choice(np.arange(1, 4), 1)[0]                                               # 随机选择1-3
            else:
                num_CAV = num_CAV                                                                               # 否则使用输入的CAV数量
            num_HDV = np.random.choice(np.arange(1, 4), 1)[0]                                                   # 随机选择1-3

        elif self.config["traffic_density"] == 2:
            # hard mode: 2-4 CAVs + 2-4 HDVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(2, 5), 1)[0]

        elif self.config["traffic_density"] == 3:
            # hard mode: 4-6 CAVs + 3-5 HDVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(4, 7), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(3, 6), 1)[0]
        self._make_vehicles(num_CAV, num_HDV)                                                                   # 创建车辆
        self.action_is_safe = True                                                                              # 动作是否安全
        self.T = int(self.config["duration"] * self.config["policy_frequency"])                                 # 时间

    def _make_road(self, ) -> None:                                                                             # 创建道路
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()                                                                                     # 道路网络

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE                                     # 线类型
        net.add_lane("a", "b", StraightLane([0, 0], [sum(self.ends[:2]), 0], line_types=[c, c]))                # 添加车道
        net.add_lane("b", "c",
                     StraightLane([sum(self.ends[:2]), 0], [sum(self.ends[:3]), 0], line_types=[c, s]))         # 添加车道
        net.add_lane("c", "d", StraightLane([sum(self.ends[:3]), 0], [sum(self.ends), 0], line_types=[c, c]))   # 添加车道

        # Merging lane
        amplitude = 3.25                                                                                        # 振幅
        ljk = StraightLane([0, 6.5 + 4], [self.ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)            # 直线车道
        lkb = SineLane(ljk.position(self.ends[0], -amplitude), ljk.position(sum(self.ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * self.ends[1]), np.pi / 2, line_types=[c, c], forbidden=True) # 正弦车道
        lbc = StraightLane(lkb.position(self.ends[1], 0), lkb.position(self.ends[1], 0) + [self.ends[2], 0],
                           line_types=[n, c], forbidden=True)                                                   # 直线车道
        net.add_lane("j", "k", ljk)                                                                             # 添加车道
        net.add_lane("k", "b", lkb)                                                                             # 添加车道
        net.add_lane("b", "c", lbc)                                                                             # 添加车道
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])     # 道路
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))                                      # 障碍物
        self.road = road                                                                                        # 道路

    def _make_vehicles(self, num_CAV=4, num_HDV=3) -> None:                                                     # 创建车辆
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road                                                                                        # 道路
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])                         # 其他车辆类型
        self.controlled_vehicles = []                                                                           # 受控车辆

        spawn_points_s = [10, 50, 90, 130, 170, 210]                                                            # 生成点
        spawn_points_m = [5, 45, 85, 125, 165, 205]                                                             # 生成点

        """Spawn points for CAV"""
        # spawn point indexes on the straight road
        spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV // 2, replace=False)                         # 生成点
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2,
                                           replace=False)                                                       # 生成点
        spawn_point_s_c = list(spawn_point_s_c)                                                                 # 生成点
        spawn_point_m_c = list(spawn_point_m_c)                                                                 # 生成点
        # remove the points to avoid duplicate
        for a in spawn_point_s_c:                                                                               # 对于每一个生成点
            spawn_points_s.remove(a)                                                                            # 删除生成点
        for b in spawn_point_m_c:                                                                               # 对于每一个生成点
            spawn_points_m.remove(b)                                                                            # 删除生成点

        """Spawn points for HDV"""
        # spawn point indexes on the straight road
        spawn_point_s_h = np.random.choice(spawn_points_s, num_HDV // 2, replace=False)                         # 生成点
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - num_HDV // 2,
                                           replace=False)                                                       # 生成点
        spawn_point_s_h = list(spawn_point_s_h)                                                                 # 生成点
        spawn_point_m_h = list(spawn_point_m_h)                                                                 # 生成点

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25  # range from [25, 27]                       # 初始速度
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5  # range from [-1.5, 1.5]                       # 位置噪声
        initial_speed = list(initial_speed)                                                                     # 转换为列表
        loc_noise = list(loc_noise)                                                                             # 转换为列表

        """spawn the CAV on the straight road first"""
        for _ in range(num_CAV // 2):                                                                           # 对于每一个CAV
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(
                spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))                      # 创建车辆
            self.controlled_vehicles.append(ego_vehicle)                                                        # 添加车辆
            road.vehicles.append(ego_vehicle)                                                                   # 添加车辆
        """spawn the rest CAV on the merging road"""
        for _ in range(num_CAV - num_CAV // 2):                                                                 # 对于每一个CAV
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))                      # 创建车辆
            self.controlled_vehicles.append(ego_vehicle)                                                        # 添加车辆
            road.vehicles.append(ego_vehicle)                                                                   # 添加车辆

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV // 2):                                                                           # 对于每一个HDV
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(                        # 创建车辆
                    spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),                                              # 位置
                                    speed=initial_speed.pop(0)))                                                # 速度

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - num_HDV // 2):                                                                 # 对于每一个HDV
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(                        # 创建车辆
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),                                              # 位置
                                    speed=initial_speed.pop(0)))                                                # 速度

    def terminate(self):                                                                                        # 终止
        return                                                                                                  # 返回

    def init_test_seeds(self, test_seeds):                                                                      # 初始化测试种子
        self.test_num = len(test_seeds)                                                                         # 测试数量
        self.test_seeds = test_seeds                                                                            # 测试种子


class MergeEnvMARL(MergeEnv):                                                                                   # 合并环境多智能体
    @classmethod
    def default_config(cls) -> dict:                                                                            # 默认配置
        config = super().default_config()                                                                       # 调用父类的默认配置
        config.update({                                                                                         # 更新配置
            "action": {                                                                                         # 动作
                "type": "MultiAgentAction",                                                                     # 多智能体动作
                "action_config": {                                                                              # 动作配置
                    "type": "DiscreteMetaAction",                                                               # 离散元动作
                    "lateral": True,                                                                            # 横向
                    "longitudinal": True                                                                        # 纵向
                }},
            "observation": {                                                                                    # 观测
                "type": "MultiAgentObservation",                                                                # 多智能体观测
                "observation_config": {                                                                         # 观测配置
                    "type": "Kinematics"                                                                        # 运动学
                }},
            "controlled_vehicles": 4                                                                            # 受控车辆
        })
        return config                                                                                           # 返回配置


register(                                                                                                       # 注册
    id='merge-v1',                                                                                              # id
    entry_point='highway_env.envs:MergeEnv',                                                                    # 入口点
)

register(                                                                                                       # 注册
    id='merge-multi-agent-v0',                                                                                  # id
    entry_point='highway_env.envs:MergeEnvMARL',                                                                # 入口点
)