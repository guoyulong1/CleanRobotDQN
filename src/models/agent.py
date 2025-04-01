import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from .dqn import DQN
from ..config import Config

class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=Config.LEARNING_RATE, use_cuda=True):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.model = DQN(state_shape, num_actions).to(self.device)
        self.target_model = DQN(state_shape, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.batch_size = Config.BATCH_SIZE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        
        self.update_target_every = 5
        self.losses = []
        self.rewards_history = []
        self.coverage_history = []
        self.steps_history = []

    def train_episode(self, env, episode, total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode}/{total_episodes}")
        print("=" * 50)
        
        while not done:
            step += 1
            action = self.act(state)
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            self.remember(state, action, reward, next_state, done)
            
            # 训练
            loss = self.replay()
            
            state = next_state
            total_reward += reward
            
            # 每100步输出一次状态
            if step % 100 == 0:
                coverage = info["coverage_rate"] * 100
                action_name = {0: "前进", 1: "左转", 2: "右转"}[action]
                print(f"步数: {step:4d} | 动作: {action_name:4s} | 覆盖率: {coverage:6.2f}% | 奖励: {reward:6.2f}")
        
        # 输出回合总结
        coverage_rate = info["coverage_rate"] * 100
        print("-" * 50)
        print(f"回合结束 - 总步数: {step}")
        print(f"最终覆盖率: {coverage_rate:.2f}%")
        print(f"总奖励: {total_reward:.2f}")
        print(f"探索率(Epsilon): {self.epsilon:.4f}")
        if loss:
            print(f"平均损失: {sum(self.losses[-100:]) / len(self.losses[-100:]):.6f}")
        print("=" * 50)
        
        return total_reward, coverage_rate, step 