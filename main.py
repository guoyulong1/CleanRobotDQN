#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CleanRobotDQN - 基于深度强化学习的机器人清扫覆盖算法
程序入口点
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import matplotlib
matplotlib.use('TkAgg')  # 改用TkAgg后端，支持交互式显示

import numpy as np
import torch
import argparse
import random

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from src.environment import CoverageEnv
from src.models import DQNAgent
from src.utils import load_map_from_image, get_available_maps, Visualizer
from src.training import train_dqn
from src.config import Config

def set_random_seed(seed=42):
    """设置随机种子，确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """主函数"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="扫地机器人全覆盖强化学习")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练机器人")
    train_parser.add_argument("--map", type=str, help="地图图像文件路径", default="maps/map.png")
    train_parser.add_argument("--episodes", type=int, help="训练回合数", default=Config.DEFAULT_EPISODES)
    train_parser.add_argument("--save-video", action="store_true", help="保存训练过程为视频")
    train_parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    train_parser.add_argument("--seed", type=int, help="随机种子", default=42)
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试机器人")
    test_parser.add_argument("--map", type=str, help="地图图像文件路径", default="maps/map.png")
    test_parser.add_argument("--save-video", action="store_true", help="保存测试过程为视频")
    test_parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    test_parser.add_argument("--seed", type=int, help="随机种子", default=42)
    
    # 列出地图命令
    subparsers.add_parser("list_maps", help="列出所有地图")
    
    args = parser.parse_args()
    
    # 根据命令执行相应操作
    if args.command == "train":
        # 设置随机种子
        set_random_seed(args.seed)
        
        # 加载地图
        map_data = load_map_from_image(args.map)
        if map_data is None:
            print(f"无法加载地图图像: {args.map}")
            return
        
        # 创建环境
        env = CoverageEnv(map_data=map_data)
        map_name = os.path.basename(args.map).split('.')[0]
        
        # 创建DQN代理
        state = env.reset()
        agent = DQNAgent(state.shape, env.action_space, use_cuda=args.gpu)
        
        try:
            # 训练
            agent = train_dqn(env, agent, args.episodes, args.save_video)
            
            # 训练完成后进行一次性测试可视化
            print("\n训练完成，开始测试可视化...")
            visualizer = Visualizer(env, map_name)
            visualizer.visualize_coverage(agent, save_video=args.save_video)
            
        finally:
            # 清理资源
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    elif args.command == "test":
        # 设置随机种子
        set_random_seed(args.seed)
        
        # 加载地图
        map_data = load_map_from_image(args.map)
        if map_data is None:
            print(f"无法加载地图图像: {args.map}")
            return
        
        # 创建环境
        env = CoverageEnv(map_data=map_data)
        map_name = os.path.basename(args.map).split('.')[0]
        
        try:
            # 加载模型
            state = env.reset()
            agent = DQNAgent(state.shape, env.action_space, use_cuda=args.gpu)
            model_path = f"models/dqn_model_{env.width}x{env.height}.h5"
            
            if not os.path.exists(model_path):
                print(f"找不到模型文件: {model_path}")
                print("请先训练模型或确认地图名称是否正确")
                return
            
            agent.load(model_path)
            print(f"加载模型: {model_path}")
            
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
        maps = get_available_maps()
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