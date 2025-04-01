import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib
matplotlib.use('TkAgg')  # 改用TkAgg后端，支持交互式显示

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
import cv2
import torch.nn.functional as F

from src.environment import CoverageEnv
from src.models.agent import DQNAgent
from src.utils.map_loader import load_map_from_image
from src.config import Config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class MapLoader:
    @staticmethod
    def load_map_from_image(image_path, target_size=(32, 32)):
        """从PNG图像加载SLAM地图数据，并调整大小到指定尺寸"""
        try:
            img = Image.open(image_path)
            img = img.resize(target_size, Image.BILINEAR)
            if img.mode != 'L':
                img = img.convert('L')
            map_data = np.array(img)
            
            # SLAM地图格式转换：
            # 黑色(0) -> 可到达区域(0)
            # 白色(255) -> 不可到达区域(1)
            # 灰色(128) -> 未知区域(2)
            map_data = np.where(map_data < 50, 0,  # 黑色区域 -> 可到达
                              np.where(map_data > 200, 1,  # 白色区域 -> 不可到达
                                     2))  # 灰色区域 -> 未知
            
            print(f"从图像加载地图: {image_path}, 原始形状: {img.size}, 调整后: {map_data.shape}")
            print(f"可到达区域: {np.sum(map_data == 0)} 像素")
            print(f"不可到达区域: {np.sum(map_data == 1)} 像素")
            print(f"未知区域: {np.sum(map_data == 2)} 像素")
            
            return map_data
        except Exception as e:
            print(f"加载地图图像失败: {e}")
            return None

    @staticmethod
    def get_available_maps(maps_dir="maps"):
        """获取maps目录下所有可用的PNG地图图像"""
        if not os.path.exists(maps_dir):
            print(f"地图目录 {maps_dir} 不存在")
            return []
        return glob.glob(os.path.join(maps_dir, "*.png"))

class Visualizer:
    def __init__(self, env, map_name):
        self.env = env
        self.map_name = map_name
        self.action_names = ["上", "下", "左", "右", "左转", "右转"]
        
    def visualize_coverage(self, agent, save_video=False):
        """可视化覆盖过程"""
        print("\n开始测试...")
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        coverage_history = []
        
        # 创建图形
        plt.ion()  # 打开交互模式
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        try:
            while not done:
                # 选择动作（测试时不使用探索）
                action = agent.act(state, training=False)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 更新状态和统计
                state = next_state
                total_reward += reward
                steps += 1
                coverage_history.append(info['coverage_rate'])
                
                # 每10步打印一次状态
                if steps % 10 == 0:
                    print(f"测试进度: 步数={steps}, 覆盖率={info['coverage_rate']:.1%}, 动作={self.action_names[action]}")
                
                # 清除当前图形
                ax.clear()
                
                # 创建地图显示
                map_display = np.zeros((self.env.height, self.env.width, 3))
                
                # 设置颜色
                map_display[self.env.accessible] = [0.8, 0.8, 0.8]  # 可到达区域为浅灰色
                map_display[self.env.obstacles] = [0.2, 0.2, 0.2]   # 障碍物为深灰色
                map_display[self.env.unknown] = [0.5, 0.5, 0.5]     # 未知区域为中灰色
                map_display[self.env.covered] = [0.0, 0.8, 0.0]     # 已覆盖区域为绿色
                
                # 标记机器人位置和方向
                y, x = self.env.robot_pos
                map_display[y, x] = [1.0, 0.0, 0.0]  # 机器人位置为红色
                
                # 标记机器人方向
                dy, dx = self.env.directions[self.env.current_direction]
                if dy is not None and dx is not None:
                    next_y, next_x = y + dy, x + dx
                    if 0 <= next_y < self.env.height and 0 <= next_x < self.env.width:
                        map_display[next_y, next_x] = [1.0, 0.5, 0.0]  # 方向指示为橙色
                
                # 绘制地图
                ax.imshow(map_display)
                ax.set_title(f"测试中...\n覆盖率: {info['coverage_rate']:.1%}\n步数: {steps}\n动作: {self.action_names[action]}")
                
                # 添加图例
                legend_elements = [
                    plt.Line2D([0], [0], marker='s', color='w', label='可到达区域', markerfacecolor='gray'),
                    plt.Line2D([0], [0], marker='s', color='w', label='障碍物', markerfacecolor='darkgray'),
                    plt.Line2D([0], [0], marker='s', color='w', label='未知区域', markerfacecolor='dimgray'),
                    plt.Line2D([0], [0], marker='s', color='w', label='已覆盖区域', markerfacecolor='green'),
                    plt.Line2D([0], [0], marker='s', color='w', label='机器人位置', markerfacecolor='red'),
                    plt.Line2D([0], [0], marker='s', color='w', label='移动方向', markerfacecolor='orange')
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
                
                # 更新显示
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)
                
                # 清理内存
                del map_display
                import gc
                gc.collect()
                
        finally:
            # 确保资源被正确释放
            plt.close(fig)
            plt.ioff()  # 关闭交互模式
            
            # 清理内存
            import gc
            gc.collect()
        
        # 输出最终统计信息
        print(f"\n测试完成!")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward:.2f}")
        print(f"最终覆盖率: {info['coverage_rate']:.1%}")

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
        
        # 确保输入通道数正确
        input_channels = input_shape[2]  # 应该是5个通道
        
        # 使用更大的卷积核和更多的通道数
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # 添加批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 计算卷积层输出大小
        conv_out_size = self._get_conv_out(input_shape)
        
        # 使用更大的全连接层
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        
        # 添加dropout层
        self.dropout = nn.Dropout(0.2)
        
    def _get_conv_out(self, shape):
        # 创建一个示例输入，确保通道数正确
        x = torch.zeros(1, shape[2], shape[0], shape[1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        # 确保输入维度正确 [batch_size, channels, height, width]
        if x.dim() == 4:
            if x.size(1) != self.conv1.in_channels:
                x = x.permute(0, 3, 1, 2)
        
        # 使用更大的批量大小
        batch_size = x.size(0)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 展平 - 使用reshape而不是view
        x = x.reshape(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

    def save(self, filename):
        """保存模型到文件"""
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        """从文件加载模型"""
        self.load_state_dict(torch.load(filename))

class DQNAgent:
    def __init__(self, state_shape, action_space, use_cuda=True):
        self.state_shape = state_shape
        self.action_space = action_space
        self.use_cuda = use_cuda
        
        # 增加内存容量和批量大小
        self.memory = deque(maxlen=200000)
        self.batch_size = 128  # 增加批量大小
        
        # 创建网络
        self.model = DQN(state_shape, action_space)
        self.target_model = DQN(state_shape, action_space)
        
        # 使用Adam优化器并调整学习率
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        # 使用GPU
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.target_model = self.target_model.to(self.device)
        else:
            self.device = torch.device("cpu")
        
        # 更新目标网络
        self.update_target_model()
        
        # 探索参数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # 降低衰减率
        
        # 训练参数
        self.gamma = 0.99
        self.update_target_every = 5  # 更频繁地更新目标网络
        
        # 预分配GPU内存
        if self.use_cuda:
            self.state_batch = torch.zeros(self.batch_size, state_shape[2], state_shape[0], state_shape[1], device=self.device)
            self.next_state_batch = torch.zeros(self.batch_size, state_shape[2], state_shape[0], state_shape[1], device=self.device)
            self.action_batch = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            self.reward_batch = torch.zeros(self.batch_size, device=self.device)
            self.done_batch = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
    
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        # 清理GPU内存
        if hasattr(self, 'device') and self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 清理模型
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'target_model'):
            del self.target_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        
        # 清理内存
        if hasattr(self, 'memory'):
            self.memory.clear()
        
        # 强制垃圾回收
        import gc
        gc.collect()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_space)
        
        # 确保输入维度正确 [batch_size, channels, height, width]
        state = np.expand_dims(state, axis=0)
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 4 and state.size(1) != self.model.conv1.in_channels:
            state = state.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    
    def replay(self):
        """训练网络"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # 使用numpy进行批量采样
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # 转换为tensor并移动到GPU，确保维度正确
        states = torch.FloatTensor(states).to(self.device)
        if states.dim() == 4 and states.size(1) != self.model.conv1.in_channels:
            states = states.permute(0, 3, 1, 2)
        
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        next_states = torch.FloatTensor(next_states).to(self.device)
        if next_states.dim() == 4 and next_states.size(1) != self.model.conv1.in_channels:
            next_states = next_states.permute(0, 3, 1, 2)
        
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（使用Double DQN）
        with torch.no_grad():
            next_q_values = self.model(next_states)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            target_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q_values
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 清理中间变量
        del states, actions, rewards, next_states, dones
        del current_q_values, next_q_values, next_actions, target_q_values
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return loss.item()
    
    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        # 清理现有资源
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'target_model'):
            del self.target_model
        if hasattr(self, 'optimizer'):
            del self.optimizer
        
        # 重新创建模型
        self.model = DQN(self.state_shape, self.action_space)
        self.target_model = DQN(self.state_shape, self.action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        # 移动到正确的设备
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.target_model = self.target_model.to(self.device)
        else:
            self.device = torch.device("cpu")
        
        # 加载状态
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # 清理加载的checkpoint
        del checkpoint
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

def train_dqn(env, episodes=1000, save_video=False):
    """训练DQN模型"""
    # 创建DQN代理
    agent = DQNAgent(env.observation_space, env.action_space)
    
    # 训练统计
    episode_rewards = []
    episode_steps = []
    episode_coverage_rates = []
    
    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # 记录每个episode的开始
        print(f"\n开始训练 Episode {episode + 1}/{episodes}")
        
        while not done:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 训练网络
            loss = agent.replay()
            
            # 更新状态
            state = next_state
            total_reward += reward
            steps += 1
            
            # 每100步输出一次训练状态
            if steps % 100 == 0:
                print(f"Episode {episode + 1}, Step {steps}, Reward: {reward:.2f}, Coverage: {info['coverage_rate']:.2%}")
        
        # 记录episode结果
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_coverage_rates.append(info['coverage_rate'])
        
        # 输出episode总结
        print(f"Episode {episode + 1} 完成:")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward:.2f}")
        print(f"覆盖率: {info['coverage_rate']:.2%}")
        
        # 每10个episode输出一次统计信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            avg_coverage = np.mean(episode_coverage_rates[-10:])
            print(f"\n最近10个episode的平均值:")
            print(f"平均步数: {avg_steps:.1f}")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"平均覆盖率: {avg_coverage:.2%}")
        
        # 更新目标网络
        if episode % agent.update_target_every == 0:
            agent.update_target_model()
    
    # 训练完成，保存模型
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = f"models/dqn_model_{env.width}x{env.height}.h5"
    agent.save(model_path)
    print(f"\n训练完成！模型已保存到: {model_path}")
    
    return agent

def main():
    parser = argparse.ArgumentParser(description="扫地机器人全覆盖强化学习")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练机器人")
    train_parser.add_argument("--map", type=str, help="地图图像文件路径", default="maps/map.png")
    train_parser.add_argument("--episodes", type=int, help="训练回合数", default=500)
    train_parser.add_argument("--save-video", action="store_true", help="保存训练过程为视频")
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试机器人")
    test_parser.add_argument("--map", type=str, help="地图图像文件路径", default="maps/map.png")
    test_parser.add_argument("--save-video", action="store_true", help="保存测试过程为视频")
    
    # 列出地图命令
    subparsers.add_parser("list_maps", help="列出所有地图")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # 加载地图
        map_data = MapLoader.load_map_from_image(args.map)
        if map_data is None:
            print(f"无法加载地图图像: {args.map}")
            return
        
        env = CoverageEnv(map_data=map_data)
        map_name = os.path.basename(args.map).split('.')[0]
        
        # 训练
        agent = train_dqn(env, args.episodes, args.save_video)
        
        # 训练完成后进行一次性测试可视化
        print("\n训练完成，开始测试可视化...")
        visualizer = Visualizer(env, map_name)
        visualizer.visualize_coverage(agent, save_video=args.save_video)
        
        # 清理资源
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    elif args.command == "test":
        # 加载地图
        map_data = MapLoader.load_map_from_image(args.map)
        if map_data is None:
            print(f"无法加载地图图像: {args.map}")
            return
        
        env = CoverageEnv(map_data=map_data)
        map_name = os.path.basename(args.map).split('.')[0]
        
        try:
            # 加载模型
            state_shape = env.reset().shape
            agent = DQNAgent(state_shape, env.action_space, use_cuda=True)
            model_path = f"models/dqn_model_{env.width}x{env.height}.h5"
            
            if not os.path.exists(model_path):
                print(f"找不到模型文件: {model_path}")
                print("请先训练模型或确认地图名称是否正确")
                return
            
            agent.load(model_path)
            
            # 可视化测试结果
            visualizer = Visualizer(env, map_name)
            visualizer.visualize_coverage(agent, save_video=args.save_video)
            
        finally:
            # 确保资源被正确释放
            if 'agent' in locals():
                del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    elif args.command == "list_maps":
        maps = MapLoader.get_available_maps()
        if maps:
            print("可用地图图像:")
            for i, map_path in enumerate(maps, 1):
                map_name = os.path.basename(map_path)
                print(f"  {i}. {map_name}")
            print(f"\n共 {len(maps)} 个地图图像文件")
        else:
            print("没有找到可用的地图图像文件。请将PNG地图图像放入maps文件夹")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()