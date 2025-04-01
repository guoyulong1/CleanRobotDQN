import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import glob
import argparse
from PIL import Image  # 添加PIL库用于图像处理

from src.environment import CoverageEnv
from src.models.agent import DQNAgent
from src.utils.map_loader import load_map_from_image
from src.config import Config

def load_map_from_image(image_path, target_size=(32, 32)):
    """从PNG图像加载地图数据，并调整大小到指定尺寸"""
    try:
        # 使用PIL库加载图像
        img = Image.open(image_path)
        
        # 调整图像大小
        img = img.resize(target_size, Image.BILINEAR)
        
        # 转换为灰度图像
        if img.mode != 'L':
            img = img.convert('L')
        
        # 转换为numpy数组
        map_data = np.array(img)
        
        # 二值化处理：小于128的像素(深色)被认为是障碍物(1)，其他为空地(0)
        map_data = np.where(map_data < 128, 1, 0)
        
        print(f"从图像加载地图: {image_path}, 原始形状: (320, 320), 调整后: {map_data.shape}")
        return map_data
    except Exception as e:
        print(f"加载地图图像失败: {e}")
        return None

def get_available_map_images(maps_dir="maps"):
    """获取maps目录下所有可用的PNG地图图像"""
    if not os.path.exists(maps_dir):
        print(f"地图目录 {maps_dir} 不存在，将创建空目录")
        os.makedirs(maps_dir)
        return []
    
    map_files = glob.glob(os.path.join(maps_dir, "*.png"))
    return map_files

class CoverageEnv:
    def __init__(self, map_data=None, width=10, height=10):
        if map_data is not None:
            self.map_data = map_data
            self.height, self.width = map_data.shape
            self.obstacles = np.where(map_data == 1, True, False)  # 1表示障碍物
        else:
            self.width = width
            self.height = height
            self.obstacles = np.zeros((height, width), dtype=bool)  # 无障碍物
        
        self.reset()

    def reset(self):
        """重置环境，初始化未清扫区域"""
        # 寻找有效的起始位置（非障碍物）
        valid_positions = [(i, j) for i in range(self.height) for j in range(self.width) 
                          if not self.obstacles[i, j]]
        if valid_positions:
            start_pos = random.choice(valid_positions)
            self.robot_pos = [start_pos[0], start_pos[1]]  # 随机选择非障碍物位置作为起点
        else:
            self.robot_pos = [0, 0]  # 如果找不到有效位置，默认为(0,0)
        
        self.covered = np.copy(self.obstacles)  # 障碍物位置视为已覆盖
        self.direction = 0  # 0: 右, 1: 下, 2: 左, 3: 上（弓字型方向）
        self.covered[self.robot_pos[0], self.robot_pos[1]] = True  # 标记起始位置为已覆盖
        self.steps = 0  # 记录步数
        return self._get_state()

    def _get_state(self):
        """返回当前状态（机器人位置 + 已清扫地图 + 障碍物）"""
        state = np.zeros((self.height, self.width, 3), dtype=np.float32)
        state[:, :, 0] = self.covered  # 通道0：已清扫区域
        state[self.robot_pos[0], self.robot_pos[1], 1] = 1  # 通道1：机器人位置
        state[:, :, 2] = self.obstacles  # 通道2：障碍物
        return state

    def step(self, action):
        """
        执行动作：
        - 0: 前进
        - 1: 左转（调整弓字型方向）
        - 2: 右转（调整弓字型方向）
        """
        reward = -0.01  # 每步小的负奖励促使机器人尽快完成任务
        done = False
        self.steps += 1

        if action == 1:  # 左转
            self.direction = (self.direction - 1) % 4
        elif action == 2:  # 右转
            self.direction = (self.direction + 1) % 4
        else:  # 前进
            x, y = self.robot_pos
            if self.direction == 0:  # 右
                y += 1
            elif self.direction == 1:  # 下
                x += 1
            elif self.direction == 2:  # 左
                y -= 1
            elif self.direction == 3:  # 上
                x -= 1

            # 检查是否越界或撞到障碍物
            if (0 <= x < self.height and 0 <= y < self.width and not self.obstacles[x, y]):
                old_pos = self.robot_pos.copy()
                self.robot_pos = [x, y]
                if not self.covered[x, y]:
                    reward = 0.5  # 清扫新区域奖励
                self.covered[x, y] = True
            else:
                reward = -0.2  # 撞墙或障碍物惩罚

        # 计算覆盖率
        coverage_rate = np.sum(self.covered) / (self.width * self.height - np.sum(self.obstacles))
        
        # 检查是否完成覆盖或达到最大步数
        if coverage_rate >= 0.95:  # 95%覆盖率视为完成
            reward += 10.0  # 完成奖励
            done = True
        elif self.steps >= self.width * self.height * 4:  # 最大步数限制
            done = True

        info = {
            "coverage_rate": coverage_rate,
            "steps": self.steps
        }
        
        return self._get_state(), reward, done, info

def create_empty_map(width, height, map_name, save_dir="maps"):
    """创建一个空地图文件"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    map_data = np.zeros((height, width), dtype=int)
    save_path = os.path.join(save_dir, f"{map_name}.txt")
    
    np.savetxt(save_path, map_data, fmt='%d')
    print(f"空地图已创建并保存至 {save_path}")
    return save_path

def create_random_obstacle_map(width, height, obstacle_ratio=0.2, map_name="random_map", save_dir="maps"):
    """创建一个带有随机障碍物的地图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    map_data = np.zeros((height, width), dtype=int)
    
    # 随机放置障碍物
    num_obstacles = int(width * height * obstacle_ratio)
    obstacle_positions = np.random.choice(width * height, num_obstacles, replace=False)
    
    for pos in obstacle_positions:
        i = pos // width
        j = pos % width
        map_data[i, j] = 1  # 1表示障碍物
    
    # 确保地图可遍历（简单检查：前后左右四个方向至少有一个不是障碍物）
    for i in range(height):
        for j in range(width):
            if map_data[i, j] == 0:  # 非障碍物位置
                neighbors = []
                if i > 0:
                    neighbors.append(map_data[i-1, j])
                if i < height-1:
                    neighbors.append(map_data[i+1, j])
                if j > 0:
                    neighbors.append(map_data[i, j-1])
                if j < width-1:
                    neighbors.append(map_data[i, j+1])
                
                if all(n == 1 for n in neighbors):  # 如果四周都是障碍物
                    # 随机清除一个障碍物
                    direction = np.random.randint(0, len(neighbors))
                    if direction == 0 and i > 0:
                        map_data[i-1, j] = 0
                    elif direction == 1 and i < height-1:
                        map_data[i+1, j] = 0
                    elif direction == 2 and j > 0:
                        map_data[i, j-1] = 0
                    elif direction == 3 and j < width-1:
                        map_data[i, j+1] = 0
    
    # 保存地图
    save_path = os.path.join(save_dir, f"{map_name}.txt")
    np.savetxt(save_path, map_data, fmt='%d')
    
    # 可视化地图
    plt.figure(figsize=(8, 8))
    plt.imshow(map_data, cmap='binary')
    plt.title(f"{map_name} - 黑色: 障碍物, 白色: 空地")
    
    # 修改这一行
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["空地", "障碍物"])  # 分开设置标签
    
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{map_name}.png"))
    plt.close()
    
    print(f"随机障碍物地图已创建并保存至 {save_path}")
    return save_path

def create_room_map(width, height, room_type="apartment", map_name=None, save_dir="maps"):
    """创建一个模拟房间的地图"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if map_name is None:
        map_name = f"{room_type}_{width}x{height}"
    
    map_data = np.zeros((height, width), dtype=int)
    
    if room_type == "apartment":
        # 创建外墙
        map_data[0, :] = 1
        map_data[height-1, :] = 1
        map_data[:, 0] = 1
        map_data[:, width-1] = 1
        
        # 创建房间隔断
        if width >= 10 and height >= 10:
            # 客厅和厨房之间的墙
            wall_pos_x = width // 3
            for i in range(1, height // 2):
                map_data[i, wall_pos_x] = 1
            
            # 厨房和卧室之间的墙
            wall_pos_y = height // 2
            for j in range(1, 2 * width // 3):
                map_data[wall_pos_y, j] = 1
            
            # 卧室和卫生间之间的墙
            wall_pos_x2 = 2 * width // 3
            for i in range(wall_pos_y + 1, height - 1):
                map_data[i, wall_pos_x2] = 1
            
            # 添加门
            map_data[height//4, wall_pos_x] = 0  # 客厅到厨房的门
            map_data[wall_pos_y, width//4] = 0  # 客厅到卧室的门
            map_data[wall_pos_y, wall_pos_x + width//6] = 0  # 厨房到卧室的门
            map_data[3*height//4, wall_pos_x2] = 0  # 卧室到卫生间的门
            
            # 添加一些家具（作为障碍物）
            # 客厅沙发
            for j in range(width//6, width//3 - 1):
                map_data[3*height//4, j] = 1
            
            # 厨房桌子
            map_data[height//4, wall_pos_x + width//10] = 1
            map_data[height//4 + 1, wall_pos_x + width//10] = 1
            
            # 卧室床
            for i in range(wall_pos_y + height//6, wall_pos_y + height//3):
                for j in range(width//3 + 2, width//2):
                    map_data[i, j] = 1
    
    elif room_type == "office":
        # 创建外墙
        map_data[0, :] = 1
        map_data[height-1, :] = 1
        map_data[:, 0] = 1
        map_data[:, width-1] = 1
        
        # 创建办公室隔断
        if width >= 12 and height >= 12:
            # 水平隔断
            for j in range(width//4, 3*width//4):
                map_data[height//3, j] = 1
                map_data[2*height//3, j] = 1
            
            # 垂直隔断
            for i in range(height//3, 2*height//3):
                map_data[i, width//3] = 1
                map_data[i, 2*width//3] = 1
            
            # 添加门
            map_data[height//3, width//2] = 0
            map_data[2*height//3, width//2] = 0
            map_data[height//2, width//3] = 0
            map_data[height//2, 2*width//3] = 0
            
            # 添加办公桌
            for i in range(height//6, height//3 - 1):
                for j in range(width//6, width//3 - 1):
                    map_data[i, j] = 1
            
            for i in range(height//6, height//3 - 1):
                for j in range(2*width//3 + 1, 5*width//6):
                    map_data[i, j] = 1
            
            for i in range(2*height//3 + 1, 5*height//6):
                for j in range(width//6, width//3 - 1):
                    map_data[i, j] = 1
            
            for i in range(2*height//3 + 1, 5*height//6):
                for j in range(2*width//3 + 1, 5*width//6):
                    map_data[i, j] = 1
    
    # 保存地图
    save_path = os.path.join(save_dir, f"{map_name}.txt")
    np.savetxt(save_path, map_data, fmt='%d')
    
    # 可视化地图
    plt.figure(figsize=(10, 10))
    plt.imshow(map_data, cmap='binary')
    plt.title(f"{map_name} - 黑色: 墙/障碍物, 白色: 空地")
    
    # 修改这一行
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["空地", "障碍物"])  # 分开设置标签
    
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{map_name}.png"))
    plt.close()
    
    print(f"{room_type}类型的地图已创建并保存至 {save_path}")
    return save_path

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[2], 16, kernel_size=8, stride=4, padding=2),  # 大幅降低分辨率
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出的特征数量
        conv_out_size = self._get_conv_out(input_shape)
        
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 128),  # 减少全连接层的神经元数量
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 128),  # 减少全连接层的神经元数量
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape).permute(0, 3, 1, 2))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        conv_out = self.conv(x).contiguous().view(x.size()[0], -1)
        
        advantage = self.advantage(conv_out)
        value = self.value(conv_out)
        
        # 使用Dueling DQN架构
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
    

class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=0.0005, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.model = DQN(state_shape, num_actions).to(self.device)
        self.target_model = DQN(state_shape, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=50000)  # 增大记忆容量
        self.batch_size = 32  # 从64减少到32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.update_target_counter = 0
        self.update_target_every = 5  # 每5个回合更新一次目标网络
        self.losses = []
        self.grad_clip_norm = 0.5  # 从1.0减小到0.5

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)  # 随机选择动作（0: 前进, 1: 左转, 2: 右转）
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 使用Double DQN：用当前模型选择动作，用目标模型评估价值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            # 通过当前模型选择最佳动作
            best_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # 通过目标模型评估这些动作的价值
            next_q_values = self.target_model(next_states).gather(1, best_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # 计算损失并更新模型
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # 使用更严格的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        
        # 在每次更新后清空不需要的中间变量，释放GPU内存
        del states, actions, rewards, next_states, dones
        del current_q_values, next_q_values, target_q_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.losses.append(loss.item())
        
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("目标网络已更新")

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"模型已保存至 {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"模型已从 {filepath} 加载")
            return True
        return False

def train_dqn(env=None, map_path=None, map_name=None, episodes=500, save_path="models"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    # 设置环境
    if env is None:
        if map_path and os.path.exists(map_path):
            map_data = load_map_from_image(map_path)
            env = CoverageEnv(map_data=map_data)
            map_name = os.path.basename(map_path).split('.')[0] if map_name is None else map_name
        else:
            env = CoverageEnv(width=10, height=10)
            map_name = "default_10x10" if map_name is None else map_name
    
    # 创建智能体
    state_shape = env.reset().shape
    num_actions = 3  # 0: 前进, 1: 左转, 2: 右转
    agent = DQNAgent(state_shape, num_actions, use_cuda=True)

    # 检查是否有已保存的模型
    model_path = os.path.join(save_path, f"dqn_agent_{map_name}.pth")
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        agent.load(model_path)
    else:
        print(f"创建新模型: {model_path}")
    
    # 训练统计
    all_rewards = []
    all_coverages = []
    all_steps = []
    all_losses = []
    
    print(f"\n{'='*50}")
    print(f"开始训练 - 地图: {map_name}, 回合数: {episodes}")
    print(f"{'='*50}\n")
    
    # 创建可视化窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Training Visualization - {map_name}")
    
    # 初始化显示区域
    coverage_map = np.zeros((env.height, env.width, 3))
    for i in range(env.height):
        for j in range(env.width):
            if env.obstacles[i, j]:
                coverage_map[i, j] = [0.5, 0.5, 0.5]  # 灰色表示障碍物
    
    im = ax1.imshow(coverage_map, interpolation='nearest')
    ax1.set_title("Coverage Map")
    ax1.grid(True)
    
    # 训练指标图
    rewards_line, = ax2.plot([], [], label='Reward')
    coverage_line, = ax2.plot([], [], label='Coverage')
    ax2.set_title("Training Metrics")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True)
    
    # 添加信息文本
    info_text = ax1.text(0.02, -0.1, '', transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    for episode in range(1, episodes+1):
        state = env.reset()
        total_reward = 0
        done = False
        episode_loss = []
        step = 0
        
        print(f"Episode {episode}/{episodes} 开始...")
        
        while not done:
            step += 1
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            
            # 更新可视化 - 每10步更新一次以提高性能
            if step % 10 == 0:
                # 更新覆盖地图
                coverage_map = np.zeros((env.height, env.width, 3))
                for i in range(env.height):
                    for j in range(env.width):
                        if env.obstacles[i, j]:
                            coverage_map[i, j] = [0.5, 0.5, 0.5]  # 灰色（障碍物）
                        elif env.covered[i, j]:
                            coverage_map[i, j] = [0.8, 0.9, 1.0]  # 浅蓝色（已覆盖）
                
                # 标记当前位置
                coverage_map[env.robot_pos[0], env.robot_pos[1]] = [0, 0, 1]  # 深蓝色（当前位置）
                
                im.set_array(coverage_map)
                
                # 更新信息文本
                coverage_rate = info["coverage_rate"] * 100
                action_names = {0: "前进", 1: "左转", 2: "右转"}
                info_text.set_text(f'Episode: {episode}/{episodes}\n'
                                 f'Step: {step} | Action: {action_names[action]}\n'
                                 f'Coverage: {coverage_rate:.1f}% | Reward: {reward:.2f}')
                
                # 更新训练指标图
                if len(all_rewards) > 0:
                    ax2.set_xlim(0, episode)
                    ax2.set_ylim(min(all_rewards) - 1, max(all_rewards) + 1)
                    rewards_line.set_data(range(len(all_rewards)), all_rewards)
                    coverage_line.set_data(range(len(all_coverages)), all_coverages)
                
                # 保存当前进度图像
                plt.savefig(f'temp/training_progress.png')
                
                # 可选：清除当前图像以释放内存
                plt.clf()
        
        # 每个回合结束后衰减探索率
        agent.decay_epsilon()
        
        # 收集训练统计数据
        all_rewards.append(total_reward)
        all_coverages.append(info["coverage_rate"] * 100)
        all_steps.append(info["steps"])
        if episode_loss:
            avg_loss = sum(episode_loss) / len(episode_loss)
            all_losses.append(avg_loss)
        
        # 打印每个回合的结果
        coverage_rate = info["coverage_rate"] * 100
        print(f"Episode {episode}/{episodes} 完成 - 步数: {step}, 覆盖率: {coverage_rate:.2f}%, 总奖励: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # 定期更新目标网络
        if episode % agent.update_target_every == 0:
            agent.update_target_model()
            print(f"目标网络已更新 (Episode {episode})")
        
        # 打印训练进度
        if episode % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / 10
            avg_coverage = sum(all_coverages[-10:]) / 10
            avg_steps = sum(all_steps[-10:]) / 10
            avg_loss = sum(all_losses[-10:]) / max(1, len(all_losses[-10:]))
            
            print(f"\n--- 训练进度 (Episodes {episode-9}-{episode}) ---")
            print(f"  平均奖励: {avg_reward:.2f}, 平均覆盖率: {avg_coverage:.2f}%")
            print(f"  平均步数: {avg_steps:.1f}, Epsilon: {agent.epsilon:.4f}, 损失: {avg_loss:.6f}")
            print(f"  记忆容量: {len(agent.memory)}/{agent.memory.maxlen}")
            print("----------------------------------------\n")
        
        # 定期保存模型
        if episode % 50 == 0 or episode == episodes:
            agent.save(model_path)
            print(f"模型已保存 (Episode {episode})")
            
            # 绘制训练过程图表
            if episode % 100 == 0 or episode == episodes:
                plot_path = os.path.join("results", f"{map_name}_training_ep{episode}.png")
                plot_training_results(all_rewards, all_coverages, all_steps, all_losses, map_name, episode)
                print(f"训练进度图表已保存至 {plot_path}")
    
    plt.ioff()
    plt.close()
    
    # 最终保存模型
    agent.save(model_path)
    
    # 绘制最终训练结果图表
    plot_training_results(all_rewards, all_coverages, all_steps, all_losses, map_name, episodes)
    
    print(f"\n{'='*50}")
    print(f"训练完成 - 地图: {map_name}, 回合数: {episodes}")
    print(f"最终覆盖率: {all_coverages[-1]:.2f}%, 最终奖励: {all_rewards[-1]:.2f}")
    print(f"模型已保存至: {model_path}")
    print(f"{'='*50}\n")
    
    return agent, map_name

def plot_training_results(rewards, coverages, steps, losses, map_name, episodes):
    """绘制训练过程中的指标变化"""
    plt.figure(figsize=(15, 10))
    
    # 绘制奖励变化
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title(f'Rewards ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # 绘制覆盖率变化
    plt.subplot(2, 2, 2)
    plt.plot(coverages)
    plt.title(f'Coverage Rate ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Coverage (%)')
    plt.ylim(0, 101)
    
    # 绘制步数变化
    plt.subplot(2, 2, 3)
    plt.plot(steps)
    plt.title(f'Steps per Episode ({map_name})')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # 绘制损失变化
    if losses:
        plt.subplot(2, 2, 4)
        plt.plot(losses)
        plt.title(f'Average Loss per Episode ({map_name})')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    
    # 保存图表
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, f"{map_name}_training_ep{episodes}.png"))
    plt.close()

def visualize_coverage(agent, env=None, map_path=None, map_name=None, save_video=False):
    """可视化机器人清扫过程"""
    plt.ion()  # 开启交互模式
    
    # 设置环境
    if env is None:
        if map_path and os.path.exists(map_path):
            map_data = load_map_from_image(map_path)
            env = CoverageEnv(map_data=map_data)
            map_name = os.path.basename(map_path).split('.')[0] if map_name is None else map_name
        else:
            env = CoverageEnv(width=10, height=10)
            map_name = "default_10x10" if map_name is None else map_name
    elif map_name is None:
        map_name = "custom_map"
    
    state = env.reset()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"Coverage Visualization - {map_name}", fontsize=16)
    
    # 设置颜色映射
    cmap = plt.cm.get_cmap('Blues', 4)
    
    # 保存每一步的图像（用于创建视频）
    frames = []
    
    # 初始状态
    coverage_map = np.zeros((env.height, env.width, 3))
    # 标记障碍物为灰色
    for i in range(env.height):
        for j in range(env.width):
            if env.obstacles[i, j]:
                coverage_map[i, j] = [0.5, 0.5, 0.5]  # 灰色
    
    # 初始位置标记为蓝色
    coverage_map[env.robot_pos[0], env.robot_pos[1]] = [0, 0, 1]  # 蓝色
    
    im = ax.imshow(coverage_map, interpolation='nearest')
    
    # 添加网格线
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(which='minor', length=0)
    
    # 绘制坐标
    for i in range(env.height):
        for j in range(env.width):
            ax.text(j, i, f'({i},{j})', va='center', ha='center', fontsize=8)
    
    # 设置信息文本
    info_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 绘制初始状态
    frames.append([im])
    
    # 执行清扫
    done = False
    step = 0
    total_reward = 0
    
    # 动作名称映射
    action_names = {0: "前进", 1: "左转", 2: "右转"}
    
    while not done:
        action = agent.act(state, training=False)  # 测试模式，不使用随机探索
        next_state, reward, done, info = env.step(action)
        state = next_state
        step += 1
        total_reward += reward
        
        # 更新可视化
        coverage_map = np.zeros((env.height, env.width, 3))
        
        # 标记已覆盖区域为浅蓝色
        for i in range(env.height):
            for j in range(env.width):
                if env.covered[i, j] and not env.obstacles[i, j]:
                    coverage_map[i, j] = [0.8, 0.9, 1.0]  # 浅蓝色
                elif env.obstacles[i, j]:
                    coverage_map[i, j] = [0.5, 0.5, 0.5]  # 灰色（障碍物）
        
        # 标记当前位置为深蓝色
        coverage_map[env.robot_pos[0], env.robot_pos[1]] = [0, 0, 1]  # 深蓝色
        
        # 更新图像
        im.set_data(coverage_map)
        
        # 更新信息文本
        coverage_rate = info["coverage_rate"] * 100
        info_text.set_text(f'步数: {step} | 动作: {action_names[action]} | 覆盖率: {coverage_rate:.1f}% | 奖励: {reward:.2f}')
        
        plt.pause(0.2)  # 暂停一小段时间，使可视化变化可见
        frames.append([im])
        
        # 如果覆盖率达到100%或者步数过多，结束模拟
        if coverage_rate >= 99.9 or step > env.width * env.height * 2:
            done = True
    
    # 显示最终结果
    plt.ioff()
    plt.title(f"Final Coverage: {coverage_rate:.2f}% | Steps: {step} | Total Reward: {total_reward:.2f}")
    
    # 如果需要保存视频
    if save_video:
        try:
            from matplotlib.animation import ArtistAnimation
            import matplotlib.animation as animation
            
            # 创建动画
            ani = ArtistAnimation(fig, frames, interval=200, blit=True)
            
            # 保存为GIF
            if not os.path.exists("videos"):
                os.makedirs("videos")
            ani.save(f"videos/{map_name}_coverage.gif", writer='pillow', fps=5)
            print(f"视频已保存至 videos/{map_name}_coverage.gif")
        except Exception as e:
            print(f"保存视频时出错: {e}")

    plt.show()

def main():
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    parser = argparse.ArgumentParser(description="扫地机器人全覆盖强化学习")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练机器人")
    train_parser.add_argument("--map", type=str, help="地图图像文件路径", default="maps/map.png")
    train_parser.add_argument("--episodes", type=int, help="训练回合数", default=500)
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试机器人")
    test_parser.add_argument("--map", type=str, help="地图图像文件路径", default="maps/map.png")
    test_parser.add_argument("--save-video", action="store_true", help="保存为视频")
    
    # 列出地图命令
    list_parser = subparsers.add_parser("list_maps", help="列出所有地图")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # 加载地图图像
        if os.path.exists(args.map):
            map_data = load_map_from_image(args.map)
            if map_data is None:
                print(f"无法加载地图图像: {args.map}")
                return
            
            env = CoverageEnv(map_data=map_data)
            map_name = os.path.basename(args.map).split('.')[0]
        else:
            print(f"地图文件不存在: {args.map}")
            print(f"使用默认地图 10x10")
            env = CoverageEnv(width=10, height=10)
            map_name = "default_10x10"
        
        agent, _ = train_dqn(env=env, map_name=map_name, episodes=args.episodes)
        visualize_coverage(agent, env=env, map_name=map_name, save_video=True)
    
    elif args.command == "test":
        # 加载地图图像
        if os.path.exists(args.map):
            map_data = load_map_from_image(args.map)
            if map_data is None:
                print(f"无法加载地图图像: {args.map}")
                return
            
            env = CoverageEnv(map_data=map_data)
            map_name = os.path.basename(args.map).split('.')[0]
        else:
            print(f"地图文件不存在: {args.map}")
            print(f"使用默认地图 10x10")
            env = CoverageEnv(width=10, height=10)
            map_name = "default_10x10"
        
        # 加载模型
        state_shape = env.reset().shape
        agent = DQNAgent(state_shape, 3, use_cuda=True)
        model_path = os.path.join("models", f"dqn_agent_{map_name}.pth")
        
        if not os.path.exists(model_path):
            print(f"找不到模型文件: {model_path}")
            print("请先训练模型或确认地图名称是否正确")
            return
        
        agent.load(model_path)
        visualize_coverage(agent, env=env, map_name=map_name, save_video=args.save_video)
    
    elif args.command == "list_maps":
        maps = get_available_map_images()
        if maps:
            print("可用地图图像:")
            for i, map_path in enumerate(maps, 1):
                map_name = os.path.basename(map_path)
                print(f"  {i}. {map_name}")
            print(f"\n共 {len(maps)} 个地图图像文件")
        else:
            print("没有找到可用的地图图像文件。请将PNG地图图像放入maps文件夹")
    
    else:
        # 默认行为：如果没有指定命令，显示帮助信息
        parser.print_help()

if __name__ == "__main__":
    main()