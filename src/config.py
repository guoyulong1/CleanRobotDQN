"""
配置文件 - 存储默认参数和配置选项
"""

class Config:
    """配置类，包含项目的默认参数"""
    
    # 环境参数
    DEFAULT_MAP_SIZE = (160, 160)
    MAX_STEPS_FACTOR = 1.5  # 最大步数 = 可到达区域数量 * 这个因子
    
    # 训练参数
    BATCH_SIZE = 32
    MEMORY_SIZE = 50000
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.998
    
    # 默认训练回合数
    DEFAULT_EPISODES = 500
    
    # 可视化参数
    VISUALIZATION_PAUSE = 0.1  # 可视化时的暂停时间
    
    # 模型参数
    CONV_FILTERS = [16, 32, 32]
    FC_UNITS = 128
    
    # 路径配置
    MAPS_DIR = "maps"
    MODELS_DIR = "models"
    VIDEOS_DIR = "videos"
    RESULTS_DIR = "results" 