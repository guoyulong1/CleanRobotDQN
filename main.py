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
        """从PNG图像加载地图数据，并调整大小到指定尺寸"""
        try:
            img = Image.open(image_path)
            img = img.resize(target_size, Image.BILINEAR)
            if img.mode != 'L':
                img = img.convert('L')
            map_data = np.array(img)
            map_data = np.where(map_data < 128, 1, 0)
            print(f"从图像加载地图: {image_path}, 原始形状: (320, 320), 调整后: {map_data.shape}")
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
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im = self.ax.imshow(np.zeros((env.height, env.width, 3)))
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                    verticalalignment='top')
        self.action_names = ['上', '下', '左', '右']
        
    def visualize_coverage(self, agent, save_video=False):
        """可视化机器人的清扫过程"""
        state = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        frames = []
        
        plt.ion()
        while not done:
            action = agent.act(state, training=False)  # 测试时不使用探索
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # 更新可视化
            coverage_map = np.zeros((self.env.height, self.env.width, 3))
            coverage_map[:, :, 0] = self.env.covered  # 已清扫区域（红色）
            coverage_map[:, :, 1] = self.env.obstacles  # 障碍物（绿色）
            coverage_map[self.env.robot_pos[0], self.env.robot_pos[1]] = [0, 0, 1]  # 机器人位置（蓝色）
            
            self.im.set_data(coverage_map)
            coverage_rate = info["coverage_rate"] * 100
            self.info_text.set_text(f'步数: {step} | 动作: {self.action_names[action]} | '
                                  f'覆盖率: {coverage_rate:.1f}% | 奖励: {reward:.2f}')
            
            plt.draw()
            plt.pause(0.2)
            if save_video:
                frames.append([self.im])
        
        plt.ioff()
        plt.title(f"最终覆盖率: {coverage_rate:.2f}% | 总步数: {step} | 总奖励: {total_reward:.2f}")
        
        if save_video:
            self._save_video(frames)
        
        plt.show()
    
    def _save_video(self, frames):
        """保存可视化过程为GIF视频"""
        try:
            from matplotlib.animation import ArtistAnimation
            if not os.path.exists("videos"):
                os.makedirs("videos")
            ani = ArtistAnimation(self.fig, frames, interval=200, blit=True)
            ani.save(f"videos/{self.map_name}_coverage.gif", writer='pillow', fps=5)
            print(f"视频已保存至 videos/{self.map_name}_coverage.gif")
        except Exception as e:
            print(f"保存视频时出错: {e}")

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
            return random.randint(0, 3)  # 随机选择动作（0: 上, 1: 下, 2: 左, 3: 右）
        
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

def train_dqn(env, map_name, episodes=500):
    """训练DQN代理"""
    state_shape = env.reset().shape
    agent = DQNAgent(state_shape, 4, use_cuda=True)
    
    # 训练统计
    episode_rewards = []
    episode_coverages = []
    episode_steps = []
    
    # 训练循环
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            state = next_state
            total_reward += reward
            steps += 1
        
        # 每个回合结束后衰减探索率
        agent.decay_epsilon()
        
        # 定期更新目标网络
        if episode % agent.update_target_every == 0:
            agent.update_target_model()
        
        # 记录训练统计
        episode_rewards.append(total_reward)
        episode_coverages.append(info["coverage_rate"] * 100)
        episode_steps.append(steps)
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_coverage = np.mean(episode_coverages[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均覆盖率: {avg_coverage:.1f}%")
            print(f"  平均步数: {avg_steps:.1f}")
            print(f"  当前探索率: {agent.epsilon:.4f}")
            print(f"  记忆容量: {len(agent.memory)}/{agent.memory.maxlen}")
            print("-" * 50)
    
    # 保存模型
    if not os.path.exists("models"):
        os.makedirs("models")
    agent.save(f"models/dqn_agent_{map_name}.pth")
    print(f"模型已保存至 models/dqn_agent_{map_name}.pth")
    
    return agent, episode_rewards

def main():
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
        agent, rewards = train_dqn(env, map_name, args.episodes)
        
        # 可视化训练结果
        visualizer = Visualizer(env, map_name)
        visualizer.visualize_coverage(agent, save_video=True)
    
    elif args.command == "test":
        # 加载地图
        map_data = MapLoader.load_map_from_image(args.map)
        if map_data is None:
            print(f"无法加载地图图像: {args.map}")
            return
        
        env = CoverageEnv(map_data=map_data)
        map_name = os.path.basename(args.map).split('.')[0]
        
        # 加载模型
        state_shape = env.reset().shape
        agent = DQNAgent(state_shape, 4, use_cuda=True)
        model_path = f"models/dqn_agent_{map_name}.pth"
        
        if not os.path.exists(model_path):
            print(f"找不到模型文件: {model_path}")
            print("请先训练模型或确认地图名称是否正确")
            return
        
        agent.load(model_path)
        
        # 可视化测试结果
        visualizer = Visualizer(env, map_name)
        visualizer.visualize_coverage(agent, save_video=args.save_video)
    
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