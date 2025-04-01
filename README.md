# DQN扫地机器人全覆盖训练项目

这是一个使用深度Q学习(DQN)算法来训练扫地机器人实现房间全覆盖的项目。机器人通过强化学习学会在房间内高效地规划清扫路径。

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── environment.py      # 环境实现
│   ├── models/            # 模型相关代码
│   └── utils/             # 工具函数
├── maps/                  # 地图文件目录
├── models/                # 训练好的模型保存目录
├── videos/                # 测试视频保存目录
├── main.py                # 主程序入口
└── requirements.txt       # 项目依赖
```

## 环境要求

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pillow

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 查看可用地图

```bash
python main.py list_maps
```

### 2. 训练机器人

基本训练命令：
```bash
python main.py train --map maps/your_map.png --episodes 500
```

参数说明：
- `--map`: 指定训练地图（默认为 maps/map.png）
- `--episodes`: 训练回合数（默认为500）

### 3. 测试机器人

基本测试命令：
```bash
python main.py test --map maps/your_map.png
```

保存测试视频：
```bash
python main.py test --map maps/your_map.png --save-video
```

参数说明：
- `--map`: 指定测试地图（默认为 maps/map.png）
- `--save-video`: 可选参数，保存测试过程为GIF视频

## 测试样例

### 样例1：使用默认地图测试
```bash
# 使用默认10x10地图训练
python main.py train --episodes 500

# 使用默认地图测试并保存视频
python main.py test --save-video
```

### 样例2：使用自定义地图
```bash
# 假设maps目录下有room.png地图
# 训练
python main.py train --map maps/room.png --episodes 1000

# 测试并保存视频
python main.py test --map maps/room.png --save-video
```

### 样例3：完整训练测试流程
```bash
# 1. 查看可用地图
python main.py list_maps

# 2. 选择地图进行训练（例如：room.png）
python main.py train --map maps/room.png --episodes 1000

# 3. 测试训练好的模型
python main.py test --map maps/room.png --save-video
```

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