import numpy as np
import random

class CoverageEnv:
    def __init__(self, map_data=None, width=10, height=10):
        if map_data is not None:
            self.map_data = map_data
            self.height, self.width = map_data.shape
            self.obstacles = np.where(map_data == 1, True, False)
        else:
            self.width = width
            self.height = height
            self.obstacles = np.zeros((height, width), dtype=bool)
        
        self.observation_space = (self.height, self.width, 3)
        self.action_space = 4  # 上、下、左、右
        self.reset()

    def reset(self):
        """重置环境状态"""
        # 初始化机器人位置（随机选择一个非障碍物位置）
        valid_positions = np.where(~self.obstacles)
        if len(valid_positions[0]) == 0:
            raise ValueError("地图中没有有效的起始位置")
        
        idx = random.randint(0, len(valid_positions[0]) - 1)
        self.robot_pos = (valid_positions[0][idx], valid_positions[1][idx])
        
        # 初始化已清扫区域
        self.covered = np.zeros((self.height, self.width), dtype=bool)
        self.covered[self.robot_pos[0], self.robot_pos[1]] = True
        
        # 计算初始覆盖率
        total_cells = np.sum(~self.obstacles)
        covered_cells = np.sum(self.covered)
        coverage_rate = covered_cells / total_cells if total_cells > 0 else 0
        
        return self._get_state()

    def step(self, action):
        """执行一步动作"""
        # 定义四个方向的移动
        directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        
        # 获取移动方向
        dy, dx = directions[action]
        
        # 计算新位置
        new_y = self.robot_pos[0] + dy
        new_x = self.robot_pos[1] + dx
        
        # 检查是否撞墙或撞到障碍物
        if (0 <= new_y < self.height and 
            0 <= new_x < self.width and 
            not self.obstacles[new_y, new_x]):
            # 移动成功
            self.robot_pos = (new_y, new_x)
            self.covered[new_y, new_x] = True
            reward = 1.0  # 移动到新位置给予奖励
        else:
            reward = -1.0  # 撞墙或撞到障碍物给予惩罚
        
        # 计算覆盖率
        total_cells = np.sum(~self.obstacles)
        covered_cells = np.sum(self.covered)
        coverage_rate = covered_cells / total_cells if total_cells > 0 else 0
        
        # 判断是否完成清扫
        done = coverage_rate >= 0.95  # 当覆盖率超过95%时认为完成
        
        # 获取新的状态
        next_state = self._get_state()
        
        # 返回下一个状态、奖励、是否完成和额外信息
        return next_state, reward, done, {
            "coverage_rate": coverage_rate,
            "robot_pos": self.robot_pos
        }

    def _get_state(self):
        """获取当前状态"""
        state = np.zeros((self.height, self.width, 3))
        state[:, :, 0] = self.covered  # 已清扫区域（红色）
        state[:, :, 1] = self.obstacles  # 障碍物（绿色）
        state[self.robot_pos[0], self.robot_pos[1]] = [0, 0, 1]  # 机器人位置（蓝色）
        return state

    # ... 其余代码保持不变 ... 