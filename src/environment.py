import numpy as np
import random
import math

class CoverageEnv:
    def __init__(self, map_data=None, width=10, height=10, robot_diameter=0.25):
        # 实际物理尺寸（米）
        self.robot_diameter = robot_diameter  # 机器人直径，单位为米
        
        if map_data is not None:
            self.map_data = map_data
            self.height, self.width = map_data.shape  # 直接使用地图的实际尺寸
            # 将地图转换为三种区域：可到达(0)、不可到达(1)、未知区域(2)
            self.accessible = np.where(map_data == 0, True, False)  # 可到达区域
            self.obstacles = np.where(map_data == 1, True, False)   # 不可到达区域
            self.unknown = np.where(map_data == 2, True, False)     # 未知区域
            
            # 根据地图尺寸计算分辨率（米/像素）
            # 160x160 对应 0.1m/像素，320x320 对应 0.05m/像素
            if max(self.height, self.width) <= 160:
                self.resolution = 0.1  # 米/像素
            else:
                self.resolution = 0.05  # 米/像素
        else:
            self.width = width
            self.height = height
            self.accessible = np.ones((height, width), dtype=bool)
            self.obstacles = np.zeros((height, width), dtype=bool)
            self.unknown = np.zeros((height, width), dtype=bool)
            self.map_data = np.zeros((height, width), dtype=int)  # 添加默认地图数据
            self.resolution = 0.1  # 默认分辨率为0.1米/像素
        
        # 计算机器人尺寸（像素）- 确保机器人半径至少为1像素
        # 对于160x160的地图，半径约为1像素
        # 对于320x320的地图，半径约为2-3像素
        self.robot_radius_pixels = max(1, int(round(self.robot_diameter / (2 * self.resolution))))
        
        # 如果可到达区域很少，就减小机器人半径
        if np.sum(self.accessible) < self.height * self.width * 0.2:
            self.robot_radius_pixels = max(1, self.robot_radius_pixels - 1)
            
        print(f"地图分辨率: {self.resolution}米/像素, 机器人半径: {self.robot_radius_pixels}像素")
        
        # 计算可到达区域的数量
        self.accessible_cells = np.sum(self.accessible)
        # 设置最大步数为可到达区域数量
        self.max_steps = self.accessible_cells*1.5
        
        # 添加连通域的计算
        self.connected_region = self._compute_connected_region()
        
        self.observation_space = (self.height, self.width, 7)  # 增加到7个通道
        self.action_space = 4  # 上、下、左、右
        self.directions = [
            (-1, 0),  # 上
            (1, 0),   # 下
            (0, -1),  # 左
            (0, 1)    # 右
        ]
        self.current_direction = 0  # 初始方向向上
        self.coverage_history = np.zeros((self.height, self.width), dtype=float)  # 覆盖历史
        self.last_positions = []  # 记录最近的位置
        self.max_history = 10  # 历史记录长度
        self.steps_taken = 0  # 记录当前步数
        
        # 新增奖惩相关变量
        self.consecutive_wall_hits = 0  # 连续撞墙次数
        self.last_action = None  # 上一次的动作
        self.action_count = np.zeros(4)  # 记录各方向动作的次数
        self.edge_cells = self._compute_edge_cells()  # 计算边界单元格
        self.frontier_cells = self._compute_frontier_cells()  # 计算前沿单元格（已覆盖与未覆盖的边界）
        
        # 新增变量：记录单元格的访问历史和重复访问计数
        self.cell_visit_count = np.zeros((self.height, self.width), dtype=int)  # 单元格访问次数
        self.last_visited_step = np.zeros((self.height, self.width), dtype=int)  # 上次访问步数
        self.repeat_visit_sequence = []  # 记录重复访问的序列
        
        self.reset()

    def reset(self):
        """重置环境状态"""
        # 初始化机器人位置（随机选择一个可到达区域）
        # 确保机器人的整个圆形范围都是可到达的
        valid_positions = self._get_valid_robot_positions()
        
        if len(valid_positions) == 0:
            # 如果找不到有效位置，简单选择一个可到达区域作为备选
            fallback_positions = self._get_simple_valid_positions()
            if len(fallback_positions) == 0:
                raise ValueError("地图中没有足够空间放置机器人")
            print("警告: 使用备选位置放置机器人")
            idx = random.randint(0, len(fallback_positions) - 1)
            self.robot_pos = fallback_positions[idx]
        else:
            idx = random.randint(0, len(valid_positions) - 1)
            self.robot_pos = valid_positions[idx]
        
        # 初始化已清扫区域
        self.covered = np.zeros((self.height, self.width), dtype=bool)
        self._mark_robot_coverage(self.robot_pos)
        
        # 初始化其他状态
        self.current_direction = 0
        self.coverage_history = np.zeros((self.height, self.width), dtype=float)
        self.last_positions = [self.robot_pos]
        
        # 计算初始覆盖率（只考虑可到达区域）
        total_accessible = np.sum(self.accessible)
        covered_cells = np.sum(self.covered & self.accessible)
        self.coverage_rate = covered_cells / total_accessible if total_accessible > 0 else 0
        
        # 在设置完初始位置后重新计算连通域
        self.connected_region = self._compute_connected_region()
        
        # 重置奖惩相关变量
        self.consecutive_wall_hits = 0
        self.last_action = None
        self.action_count = np.zeros(4)
        self.frontier_cells = self._compute_frontier_cells()
        
        # 重置访问历史和重复访问计数
        self.cell_visit_count = np.zeros((self.height, self.width), dtype=int)
        self.last_visited_step = np.zeros((self.height, self.width), dtype=int)
        self.repeat_visit_sequence = []
        
        # 标记初始位置已访问
        y, x = self.robot_pos
        self.cell_visit_count[y, x] = 1
        self.last_visited_step[y, x] = 0
        
        self.steps_taken = 0  # 重置步数
        return self._get_state()

    def step(self, action):
        """执行动作"""
        self.steps_taken += 1
        dy, dx = self.directions[action]
        new_y = self.robot_pos[0] + dy
        new_x = self.robot_pos[1] + dx
        new_pos = (new_y, new_x)
        
        # 记录动作统计
        self.action_count[action] += 1
        
        # 检查新位置是否有效
        # 简化检查，提高效率
        if (new_y < 0 or new_y >= self.height or 
            new_x < 0 or new_x >= self.width or 
            not self.accessible[new_y, new_x]):
            # 4. 优化撞墙惩罚: 连续撞墙惩罚递增
            self.consecutive_wall_hits += 1
            wall_penalty = -2.0 - 0.5 * min(5, self.consecutive_wall_hits - 1)  # 最多增加到-4.5
            reward = wall_penalty
            done = False
            info = {
                'coverage_rate': self.coverage_rate,
                'hit_wall': True,
                'steps': self.steps_taken
            }
            return self._get_state(), reward, done, info
        
        # 成功移动，重置连续撞墙计数
        self.consecutive_wall_hits = 0
        
        # 更新位置
        prev_pos = self.robot_pos
        self.robot_pos = new_pos
        self.last_positions.append(self.robot_pos)
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)
        
        # 更新单元格访问信息
        y, x = self.robot_pos
        self.cell_visit_count[y, x] += 1
        visit_count = self.cell_visit_count[y, x]
        steps_since_last_visit = self.steps_taken - self.last_visited_step[y, x]
        self.last_visited_step[y, x] = self.steps_taken
        
        # 更新覆盖区域并计算新覆盖的像素数
        prev_covered_count = np.sum(self.covered & self.accessible)
        self._mark_robot_coverage(self.robot_pos)
        new_covered_count = np.sum(self.covered & self.accessible)
        newly_covered = new_covered_count - prev_covered_count
        
        # 重新计算前沿单元格
        self.frontier_cells = self._compute_frontier_cells()
        
        # 计算基础奖励
        reward = 0
        
        # 1. 优化覆盖奖励: 根据新覆盖区域给予奖励
        if newly_covered > 0:
            # 更新覆盖率
            self.coverage_rate = new_covered_count / np.sum(self.accessible)
            # 新区域覆盖奖励 - 增加奖励力度
            coverage_reward = newly_covered * (1.0 + 2.0 * self.coverage_rate)  # 提高基础奖励和覆盖率比例
            reward += coverage_reward
            
            # 重置重复访问序列
            self.repeat_visit_sequence = []
        else:
            # 2. 加强重复清扫惩罚 - 根据访问次数动态增加惩罚
            if visit_count > 1:
                # 添加到重复访问序列
                if len(self.repeat_visit_sequence) >= 5:
                    self.repeat_visit_sequence.pop(0)
                self.repeat_visit_sequence.append(new_pos)
                
                # 动态递增惩罚：基础-0.5，每多访问一次增加50%
                repeat_penalty = -0.5 * (min(5, visit_count - 1) ** 1.5)
                
                # 检查是否形成复杂的重复模式
                if self._is_repetitive_movement():
                    repeat_penalty *= 1.5  # 更严重的重复模式惩罚更重
                
                reward += repeat_penalty
            else:
                # 新区域但无新覆盖，无惩罚
                pass
        
        # 3. 增加方向一致性奖励
        if self.last_action == action:
            # 如果保持同一方向移动，给予小额奖励
            reward += 0.1
        self.last_action = action
        
        # 5. 显著增加边界和前沿区域探索奖励
        is_near_frontier = self._is_near_frontier()
        if is_near_frontier:
            # 大幅提高前沿区域奖励
            reward += 1.0
        
        if self._is_near_edge(self.robot_pos):
            # 靠近边界区域
            reward += 0.5
        
        # 1.b 增加对未探索区域的奖励
        if self._has_nearby_uncovered():
            # 如果周围有未覆盖区域，提高额外奖励
            reward += 0.8  # 从0.3增加到0.8
        
        # 6. 动态调整步数惩罚 - 低覆盖率时取消惩罚
        if self.coverage_rate < 0.3:
            # 低覆盖率时不惩罚，鼓励更多探索
            step_penalty = 0.0
        elif self.coverage_rate < 0.7:
            # 中等覆盖率时轻微惩罚
            step_penalty = -0.05
        else:
            # 高覆盖率时适度惩罚，鼓励尽快完成
            step_penalty = -0.15
        reward += step_penalty
        
        # 修改终止条件
        done = (self.coverage_rate >= 0.95 or  # 达到目标覆盖率
                self.steps_taken >= self.max_steps)  # 超过最大步数
        
        # 如果达到目标覆盖率，给予额外奖励
        if self.coverage_rate >= 0.95:
            # 根据步数调整完成奖励，步数越少奖励越高
            completion_bonus = 15.0 + max(0, (self.max_steps - self.steps_taken) / self.max_steps * 10.0)
            reward += completion_bonus
        
        info = {
            'coverage_rate': self.coverage_rate,
            'hit_wall': False,
            'steps': self.steps_taken,
            'newly_covered': newly_covered,
            'visit_count': visit_count,
            'near_frontier': is_near_frontier
        }
        
        return self._get_state(), reward, done, info

    def _get_simple_valid_positions(self):
        """获取简单的有效位置列表（仅检查中心点是否可到达）"""
        valid_positions = []
        # 使用numpy获取可到达区域的坐标
        accessible_points = np.where(self.accessible)
        # 将坐标组合成点列表201711
        for i in range(len(accessible_points[0])):
            y, x = accessible_points[0][i], accessible_points[1][i]
            valid_positions.append((y, x))
        return valid_positions

    def _compute_edge_cells(self):
        """计算地图边界和障碍物边缘单元格"""
        edge_cells = np.zeros((self.height, self.width), dtype=bool)
        
        # 地图边界
        edge_cells[0, :] = True
        edge_cells[self.height-1, :] = True
        edge_cells[:, 0] = True
        edge_cells[:, self.width-1] = True
        
        # 障碍物边缘 (可到达区域与障碍物的边界)
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.accessible[y, x]:
                    # 检查这个可到达单元格是否与障碍物相邻
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.height and 0 <= nx < self.width and 
                            self.obstacles[ny, nx]):
                            edge_cells[y, x] = True
                            break
        
        return edge_cells
    
    def _compute_frontier_cells(self):
        """计算前沿单元格（已覆盖与未覆盖的边界）"""
        frontier_cells = np.zeros((self.height, self.width), dtype=bool)
        
        # 如果没有已覆盖区域，返回空数组
        if not hasattr(self, 'covered'):
            return frontier_cells
        
        # 寻找已覆盖区域与未覆盖区域的边界
        for y in range(self.height):
            for x in range(self.width):
                if self.accessible[y, x] and not self.covered[y, x]:
                    # 检查这个未覆盖单元格是否与已覆盖区域相邻
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.height and 0 <= nx < self.width and 
                            self.covered[ny, nx]):
                            frontier_cells[y, x] = True
                            break
        
        return frontier_cells
    
    def _is_near_edge(self, pos):
        """检查是否靠近地图边界或障碍物边缘"""
        y, x = pos
        # 检查pos及其周围是否包含边界单元格
        check_radius = self.robot_radius_pixels + 1
        
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.edge_cells[ny, nx]):
                    return True
        return False
    
    def _is_near_frontier(self):
        """检查是否靠近前沿区域"""
        y, x = self.robot_pos
        # 检查机器人周围是否包含前沿单元格
        check_radius = self.robot_radius_pixels + 2
        
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.frontier_cells[ny, nx]):
                    return True
        return False

    def _get_valid_robot_positions(self):
        """获取可以放置机器人的有效位置列表"""
        # 优化：不检查所有点，只检查部分点
        valid_positions = []
        sample_step = max(1, min(self.height, self.width) // 50)  # 每隔sample_step个点采样一次
        
        # 改成更高效的实现
        for y in range(0, self.height, sample_step):
            for x in range(0, self.width, sample_step):
                if self._is_valid_robot_position((y, x)):
                    valid_positions.append((y, x))
                    # 找到足够多的点后停止
                    if len(valid_positions) >= 10:
                        return valid_positions
        
        return valid_positions

    def _is_valid_robot_position(self, pos):
        """检查给定位置是否可以放置机器人（整个圆形区域都是可到达的）"""
        y, x = pos
        
        # 检查中心点是否在地图范围内并且可到达
        if (y < 0 or y >= self.height or x < 0 or x >= self.width or
            not self.accessible[y, x]):
            return False
        
        # 对于半径为1的情况，只检查中心点和相邻点
        if self.robot_radius_pixels <= 1:
            # 检查中心点周围的四个相邻点
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    not self.accessible[ny, nx]):
                    return False
            return True
        
        # 对于较大半径，检查圆形范围内的关键点
        # 检查半径上的点而不是所有点，提高效率
        for angle in range(0, 360, 45):  # 每45度检查一个点
            rad = math.radians(angle)
            dy = int(round(self.robot_radius_pixels * math.sin(rad)))
            dx = int(round(self.robot_radius_pixels * math.cos(rad)))
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < self.height and 0 <= nx < self.width):
                if not self.accessible[ny, nx] or self.obstacles[ny, nx]:
                    return False
            else:
                return False  # 超出地图范围
        
        return True

    def _mark_robot_coverage(self, pos):
        """标记机器人覆盖的区域"""
        y, x = pos
        
        # 标记中心点
        if 0 <= y < self.height and 0 <= x < self.width and self.accessible[y, x]:
            self.covered[y, x] = True
            
        # 如果半径为1或更小，只标记中心和相邻点
        if self.robot_radius_pixels <= 1:
            for dy, dx in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and 
                    self.accessible[ny, nx]):
                    self.covered[ny, nx] = True
            return
            
        # 标记圆形范围内的所有可到达点为已覆盖
        for dy in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
            for dx in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
                # 计算到中心的距离
                distance = math.sqrt(dy**2 + dx**2)
                if distance <= self.robot_radius_pixels:
                    new_y, new_x = y + dy, x + dx
                    # 检查点是否在地图范围内且是可到达区域
                    if (0 <= new_y < self.height and 
                        0 <= new_x < self.width and 
                        self.accessible[new_y, new_x]):
                        self.covered[new_y, new_x] = True

    def _get_state(self):
        """获取当前状态"""
        # 创建状态表示 - 增加到7个通道
        state = np.zeros((self.height, self.width, 7))
        
        # 通道1: 可到达区域
        state[:, :, 0] = self.accessible.astype(float)
        
        # 通道2: 障碍物
        state[:, :, 1] = self.obstacles.astype(float)
        
        # 通道3: 未知区域
        state[:, :, 2] = self.unknown.astype(float)
        
        # 通道4: 机器人位置和形状
        y, x = self.robot_pos
        # 标记圆形机器人（而不仅仅是中心点）
        for dy in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
            for dx in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
                # 计算到中心的距离
                distance = math.sqrt(dy**2 + dx**2)
                if distance <= self.robot_radius_pixels:
                    new_y, new_x = y + dy, x + dx
                    # 检查点是否在地图范围内
                    if 0 <= new_y < self.height and 0 <= new_x < self.width:
                        state[new_y, new_x, 3] = 1
        
        # 通道5: 方向信息
        state[:, :, 4] = self._get_direction_map()
        
        # 新增通道6: 已覆盖区域
        state[:, :, 5] = self.covered.astype(float)
        
        # 新增通道7: 未覆盖的可达区域
        state[:, :, 6] = (~self.covered & self.accessible).astype(float)
        
        return state

    def _get_direction_map(self):
        """获取方向信息图"""
        direction_map = np.zeros((self.height, self.width))
        y, x = self.robot_pos
        dy, dx = self.directions[self.current_direction]
        
        # 在机器人前方标记方向，考虑机器人大小
        start_dist = self.robot_radius_pixels + 1  # 从机器人边缘开始
        for i in range(start_dist, start_dist + 3):  # 向前看3格
            next_y, next_x = y + dy * i, x + dx * i
            # 标记方向点及其周围区域
            for d_y in range(-1, 2):
                for d_x in range(-1, 2):
                    check_y, check_x = next_y + d_y, next_x + d_x
                    if 0 <= check_y < self.height and 0 <= check_x < self.width:
                        direction_map[check_y, check_x] = 1
        
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

    def _compute_connected_region(self):
        """计算起始位置所在的连通域"""
        connected = np.zeros((self.height, self.width), dtype=bool)
        if not hasattr(self, 'robot_pos'):
            return connected
        
        # 使用BFS计算连通区域
        queue = [self.robot_pos]
        connected[self.robot_pos[0], self.robot_pos[1]] = True
        
        while queue:
            current = queue.pop(0)
            for dy, dx in self.directions:
                new_y, new_x = current[0] + dy, current[1] + dx
                if (0 <= new_y < self.height and 
                    0 <= new_x < self.width and 
                    self.accessible[new_y, new_x] and 
                    not connected[new_y, new_x]):
                    connected[new_y, new_x] = True
                    queue.append((new_y, new_x))
        
        return connected

    def _is_repetitive_movement(self):
        """检查是否存在重复移动模式"""
        if len(self.last_positions) < 4:
            return False
        
        # 检查最近4步是否形成来回移动
        last_4 = np.array(self.last_positions[-4:])
        if (np.array_equal(last_4[0], last_4[2]) and 
            np.array_equal(last_4[1], last_4[3])):
            return True
        return False

    def _has_nearby_uncovered(self):
        """检查周围是否有未覆盖的可达区域"""
        y, x = self.robot_pos
        
        # 扩大检查范围，考虑机器人尺寸
        check_radius = self.robot_radius_pixels + 2
        
        for dy in range(-check_radius, check_radius + 1):
            for dx in range(-check_radius, check_radius + 1):
                new_y, new_x = y + dy, x + dx
                if (0 <= new_y < self.height and 
                    0 <= new_x < self.width and 
                    self.accessible[new_y, new_x] and 
                    not self.covered[new_y, new_x]):
                    return True
        return False

    # ... 其余代码保持不变 ... 