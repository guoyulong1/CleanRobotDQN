# CleanRobotDQN

基于深度强化学习的机器人清扫覆盖算法

## 项目简介

这个项目使用深度强化学习（DQN）训练一个能够自主清扫覆盖整个区域的机器人。机器人在一个包含障碍物的环境中学习如何有效地覆盖所有可达区域，同时避免障碍物和重复清扫已覆盖区域。

### 主要特点

- 使用深度Q网络（DQN）进行强化学习
- 支持各种尺寸和复杂度的地图
- 动态奖励机制，优化探索和覆盖率
- 可视化测试结果
- 优先经验回放和动态探索率

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 2.0.0+
- NumPy 1.19.5+
- Matplotlib 3.5.0+
- OpenCV 4.5.3+

### 安装步骤

1. 克隆仓库：
   ```
   git clone https://github.com/yourusername/CleanRobotDQN.git
   cd CleanRobotDQN
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

### 列出可用地图

```bash
python main.py list_maps
```

### 训练机器人

```bash
python main.py train --map maps/map.png --episodes 500 --gpu
```

参数说明：
- `--map`: 地图图像路径
- `--episodes`: 训练回合数
- `--gpu`: 使用GPU加速训练
- `--save-video`: 保存训练过程视频
- `--seed`: 随机种子

### 测试机器人

```bash
python main.py test --map maps/map.png --gpu
```

参数说明：
- `--map`: 地图图像路径
- `--gpu`: 使用GPU加速测试
- `--save-video`: 保存测试过程视频
- `--seed`: 随机种子

## 自定义地图

1. 准备一个PNG图像作为地图：
   - 黑色区域 (RGB 0,0,0): 可到达区域
   - 白色区域 (RGB 255,255,255): 障碍物
   - 灰色区域 (RGB 128,128,128): 未知区域

2. 将地图图像放入 `maps` 目录

3. 使用新地图进行训练或测试：
   ```bash
   python main.py train --map maps/your_map.png
   ```

## 项目结构

```
CleanRobotDQN/
├── main.py                 # 程序入口点
├── requirements.txt        # 项目依赖
├── maps/                   # 地图文件夹
│   └── map.png             # 默认地图
├── models/                 # 保存训练好的模型
└── src/                    # 源代码
    ├── environment.py      # 环境定义
    ├── config.py           # 配置文件
    ├── models/             # 网络和代理定义
    │   ├── network.py      # DQN网络架构
    │   └── agent.py        # DQN代理类
    ├── training/           # 训练相关功能
    │   └── trainer.py      # 训练器
    └── utils/              # 工具函数
        ├── map_loader.py   # 地图加载器
        └── visualizer.py   # 可视化工具
```

## 主要参数配置

主要配置参数位于 `src/config.py` 文件中，可以根据需要进行调整：

- `DEFAULT_MAP_SIZE`: 默认地图大小
- `MAX_STEPS_FACTOR`: 最大步数因子（可达区域数量 * 因子）
- `EPSILON_START`: 初始探索率
- `EPSILON_MIN`: 最小探索率
- `EPSILON_DECAY`: 探索率衰减

## 注意事项

1. 训练前确保：
   - maps目录中有对应的地图文件（PNG格式）
   - 地图文件为二值图像（黑色表示障碍物，白色表示可通行区域）

2. 测试前确保：
   - models目录中有对应的训练好的模型文件
   - 模型文件名格式为：dqn_agent_[地图名].pth

3. 视频输出：
   - 测试视频将保存在videos目录下
   - 视频格式为GIF，文件名格式为：[地图名]_coverage.gif

## 训练效果

- 机器人会学习到高效的清扫路径
- 训练过程中会显示实时覆盖率
- 测试时会可视化机器人的清扫过程
- 可以通过视频回放分析机器人的清扫策略

## 常见问题

1. 如果遇到"找不到模型文件"错误：
   - 确保已经完成训练
   - 检查地图名称是否正确
   - 确认模型文件存在于models目录

2. 如果地图加载失败：
   - 确保地图文件为PNG格式
   - 检查地图文件是否损坏
   - 确认地图文件路径正确

3. 如果训练效果不理想：
   - 增加训练回合数
   - 调整奖励参数
   - 尝试不同的地图大小和复杂度

## 许可证

MIT License