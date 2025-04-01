import numpy as np
import random

class CoverageEnv:
    def __init__(self, map_data=None, width=10, height=10):
        if map_data is not None:
            self.map_data = map_data
            self.height, self.width = map_data.shape
            # 将地图转换为三种区域：可到达(0)、不可到达(1)、未知区域(2)
            self.accessible = np.where(map_data == 0, True, False)  # 可到达区域
            self.obstacles = np.where(map_data == 1, True, False)   # 不可到达区域
            self.unknown = np.where(map_data == 2, True, False)     # 未知区域
        else:
            self.width = width
            self.height = height
            self.accessible = np.ones((height, width), dtype=bool)
            self.obstacles = np.zeros((height, width), dtype=bool)
            self.unknown = np.zeros((height, width), dtype=bool)
        
        self.observation_space = (self.height, self.width, 5)  # 增加状态通道
        self.action_space = 6  # 上、下、左、右、左转90度、右转90度
        self.directions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
            4: None,     # 左转90度
            5: None      # 右转90度
        }
        self.current_direction = 0  # 初始方向向上
        self.coverage_history = np.zeros((self.height, self.width), dtype=float)  # 覆盖历史
        self.last_positions = []  # 记录最近的位置
        self.max_history = 10  # 历史记录长度
        self.reset()

    def reset(self):
        """重置环境状态"""
        # 初始化机器人位置（随机选择一个可到达区域）
        valid_positions = np.where(self.accessible)
        if len(valid_positions[0]) == 0:
            raise ValueError("地图中没有有效的可到达区域")
        
        idx = random.randint(0, len(valid_positions[0]) - 1)
        self.robot_pos = (valid_positions[0][idx], valid_positions[1][idx])
        
        # 初始化已清扫区域
        self.covered = np.zeros((self.height, self.width), dtype=bool)
        self.covered[self.robot_pos[0], self.robot_pos[1]] = True
        
        # 初始化其他状态
        self.current_direction = 0
        self.coverage_history = np.zeros((self.height, self.width), dtype=float)
        self.last_positions = [self.robot_pos]
        
        # 计算初始覆盖率（只考虑可到达区域）
        total_accessible = np.sum(self.accessible)
        covered_cells = np.sum(self.covered & self.accessible)
        self.coverage_rate = covered_cells / total_accessible if total_accessible > 0 else 0
        
        return self._get_state()

    def step(self, action):
        """执行一步动作"""
        # 获取当前方向
        current_direction = self.current_direction
        
        # 处理转向动作
        if action in [4, 5]:  # 左转或右转
            if action == 4:  # 左转
                self.current_direction = (self.current_direction - 1) % 4
            else:  # 右转
                self.current_direction = (self.current_direction + 1) % 4
            reward = -0.1  # 转向的惩罚
            done = False
            info = {'coverage_rate': self.coverage_rate}
            return self._get_state(), reward, done, info
        
        # 处理移动动作
        dy, dx = self.directions[action]
        new_y, new_x = self.robot_pos[0] + dy, self.robot_pos[1] + dx
        
        # 检查是否撞墙或超出边界
        if (new_y < 0 or new_y >= self.height or 
            new_x < 0 or new_x >= self.width or 
            not self.accessible[new_y, new_x]):
            reward = -1.0  # 撞墙的惩罚
            done = False
            info = {'coverage_rate': self.coverage_rate}
            return self._get_state(), reward, done, info
        
        # 更新机器人位置
        self.robot_pos = (new_y, new_x)
        
        # 更新位置历史
        self.last_positions.append(self.robot_pos)
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)
        
        # 检查是否覆盖新区域
        if not self.covered[new_y, new_x]:
            self.covered[new_y, new_x] = True
            self.coverage_rate = np.sum(self.covered) / np.sum(self.accessible)
            reward = 1.0  # 覆盖新区域的奖励
        else:
            reward = -0.2  # 重复覆盖的惩罚
        
        # 检查是否完成覆盖
        done = self.coverage_rate >= 0.90
        
        # 额外的奖励/惩罚机制
        # 1. 奖励系统性的覆盖模式
        if self._is_systematic_coverage():
            reward += 0.5
        
        # 2. 奖励保持直线移动
        if action == current_direction:
            reward += 0.3
        
        # 3. 惩罚频繁改变方向
        if action != current_direction:
            reward -= 0.1
        
        # 4. 奖励向未覆盖区域移动
        if self._is_moving_towards_uncovered():
            reward += 0.2
        
        # 5. 惩罚在原地徘徊
        if self._is_staying_in_same_area():
            reward -= 0.3
        
        info = {'coverage_rate': self.coverage_rate}
        return self._get_state(), reward, done, info

    def _is_moving_towards_uncovered(self):
        """检查是否在向未覆盖区域移动"""
        y, x = self.robot_pos
        dy, dx = self.directions[self.current_direction]
        next_y, next_x = y + dy, x + dx
        
        if 0 <= next_y < self.height and 0 <= next_x < self.width:
            # 检查前方3个格子的覆盖情况
            for i in range(1, 4):
                check_y, check_x = y + dy * i, x + dx * i
                if 0 <= check_y < self.height and 0 <= check_x < self.width:
                    if not self.covered[check_y, check_x]:
                        return True
        return False

    def _is_staying_in_same_area(self):
        """检查是否在原地徘徊"""
        if len(self.last_positions) < 5:
            return False
        
        # 检查最近5个位置是否都在3x3的区域内
        recent_positions = self.last_positions[-5:]
        min_y = min(y for y, _ in recent_positions)
        max_y = max(y for y, _ in recent_positions)
        min_x = min(x for _, x in recent_positions)
        max_x = max(x for _, x in recent_positions)
        
        return (max_y - min_y <= 2) and (max_x - min_x <= 2)

    def _get_state(self):
        """获取当前状态"""
        # 创建状态表示
        state = np.zeros((self.height, self.width, 5))
        
        # 通道1: 可到达区域
        state[:, :, 0] = self.accessible.astype(float)
        
        # 通道2: 障碍物
        state[:, :, 1] = self.obstacles.astype(float)
        
        # 通道3: 未知区域
        state[:, :, 2] = self.unknown.astype(float)
        
        # 通道4: 机器人位置
        y, x = self.robot_pos
        state[y, x, 3] = 1
        
        # 通道5: 方向信息
        state[:, :, 4] = self._get_direction_map()
        
        return state

    def _get_direction_map(self):
        """获取方向信息图"""
        direction_map = np.zeros((self.height, self.width))
        y, x = self.robot_pos
        dy, dx = self.directions[self.current_direction]
        
        # 在机器人前方标记方向
        for i in range(1, 4):  # 向前看3格
            next_y, next_x = y + dy * i, x + dx * i
            if 0 <= next_y < self.height and 0 <= next_x < self.width:
                direction_map[next_y, next_x] = 1
        
        return direction_map

    def _is_systematic_coverage(self):
        """检查是否形成系统覆盖模式"""
        # 检查是否在形成直线或规则的覆盖模式
        if len(self.last_positions) < 3:
            return False
        
        # 检查最近的位置是否形成直线
        positions = np.array(self.last_positions[-3:])
        if np.all(positions[:, 0] == positions[0, 0]) or np.all(positions[:, 1] == positions[0, 1]):
            return True
        
        # 检查是否在形成规则的转向模式
        if len(self.last_positions) >= 5:
            positions = np.array(self.last_positions[-5:])
            if self._is_zigzag_pattern(positions):
                return True
        
        return False

    def _is_straight_line(self):
        """检查是否保持直线运动"""
        if len(self.last_positions) < 3:
            return False
        positions = np.array(self.last_positions[-3:])
        return np.all(positions[:, 0] == positions[0, 0]) or np.all(positions[:, 1] == positions[0, 1])

    def _is_zigzag_pattern(self, positions):
        """检查是否形成之字形模式"""
        # 检查y坐标是否交替变化
        y_changes = np.diff(positions[:, 0])
        if len(y_changes) >= 2:
            if np.all(y_changes[::2] == y_changes[0]) and np.all(y_changes[1::2] == -y_changes[0]):
                return True
        return False

    # ... 其余代码保持不变 ... 