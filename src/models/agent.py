import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gc

from .network import DQN

class DQNAgent:
    """DQN智能体类，负责决策、学习和记忆"""
    
    def __init__(self, state_shape, action_space, use_cuda=True):
        """初始化DQN智能体
        
        Args:
            state_shape: 状态空间形状 (height, width, channels)
            action_space: 动作空间大小
            use_cuda: 是否使用GPU
        """
        self.state_shape = state_shape
        self.action_space = action_space
        self.use_cuda = use_cuda
        
        # 经验回放和批量设置
        self.memory = deque(maxlen=50000)  # 经验回放缓冲区
        self.batch_size = 32  # 批量大小
        self.priority_memory = deque(maxlen=5000)  # 优先经验回放缓冲区
        
        # 创建网络
        self.model = DQN(state_shape, action_space)
        self.target_model = DQN(state_shape, action_space)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        # 设备配置
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.target_model = self.target_model.to(self.device)
        else:
            self.device = torch.device("cpu")
        
        # 更新目标网络
        self.update_target_model()
        
        # 探索参数设置
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 最小探索率
        self.epsilon_decay = 0.998  # 探索率衰减
        self.coverage_rate = 0.0  # 用于动态调整探索率
        
        # 学习参数
        self.gamma = 0.99  # 折扣因子
        self.update_target_every = 5  # 每5个episode更新一次目标网络
        
        # 记录训练状态
        self.training_episodes = 0
        
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
        if hasattr(self, 'priority_memory'):
            self.priority_memory.clear()
        
        # 强制垃圾回收
        gc.collect()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
        # 判断是否是高价值经验（高奖励或惩罚）
        if abs(reward) > 1.0 or done:
            # 高奖励或任务完成的经验优先记忆
            self.priority_memory.append(experience)
    
    def act(self, state, training=True, info=None):
        """选择动作
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
            info: 环境信息字典
            
        Returns:
            action: 选择的动作
        """
        # 更新覆盖率（如果info中提供）
        if info and 'coverage_rate' in info:
            self.coverage_rate = info['coverage_rate']
        
        # 动态探索率：覆盖率低时增加探索，覆盖率高时减少探索
        dynamic_epsilon = self.epsilon
        if training and self.coverage_rate > 0:
            # 覆盖率在80%以上时，基于覆盖率动态调整探索率
            if self.coverage_rate >= 0.8:
                # 覆盖率高时，降低探索率以专注于优化路径
                dynamic_epsilon = max(self.epsilon_min, self.epsilon * (1.0 - self.coverage_rate))
            elif self.coverage_rate < 0.3:
                # 覆盖率低时，保持较高探索率
                dynamic_epsilon = max(self.epsilon, 0.3)
        
        if training and random.random() < dynamic_epsilon:
            # 随机探索
            if info and 'near_frontier' in info and info['near_frontier'] and random.random() < 0.7:
                # 如果在前沿区域附近，偏向选择未尝试过的动作
                valid_actions = list(range(self.action_space))
                last_action = info.get('last_action', None)
                if last_action is not None and random.random() < 0.5:
                    # 50%概率避免选择上一个动作（减少原地踏步）
                    if last_action in valid_actions:
                        valid_actions.remove(last_action)
                return random.choice(valid_actions)
            return random.randrange(self.action_space)
        
        # 模型决策模式
        # 确保输入维度正确 [batch_size, channels, height, width]
        state = np.expand_dims(state, axis=0)
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 4 and state.size(1) != self.model.conv1.in_channels:
            state = state.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    
    def replay(self):
        """从经验回放缓冲区中学习
        
        Returns:
            loss: 训练损失值
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # 混合采样：从普通记忆和优先记忆中采样
        priority_ratio = 0.3  # 优先经验的比例
        priority_size = min(int(self.batch_size * priority_ratio), len(self.priority_memory))
        normal_size = self.batch_size - priority_size
        
        batch = []
        if priority_size > 0:
            # 从优先记忆中采样
            priority_batch = random.sample(self.priority_memory, priority_size)
            batch.extend(priority_batch)
        
        if normal_size > 0:
            # 从普通记忆中采样
            normal_batch = random.sample(self.memory, normal_size)
            batch.extend(normal_batch)
        
        # 如果覆盖率较低，增加特殊样本以鼓励探索
        if self.coverage_rate < 0.5 and len(batch) == self.batch_size:
            # 增加对有奖励样本的重视
            rewards = [abs(x[2]) for x in batch]
            min_reward_idx = rewards.index(min(rewards))
            
            # 从记忆中选择一个高奖励样本替换低奖励样本
            for i, experience in enumerate(self.memory):
                if abs(experience[2]) > 1.0:  # 高奖励
                    batch[min_reward_idx] = experience
                    break
        
        # 处理批量数据
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
        """保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """加载模型
        
        Args:
            filepath: 模型路径
        """
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