class Config:
    # 环境参数
    DEFAULT_MAP_SIZE = (32, 32)
    MAX_STEPS_FACTOR = 4  # 最大步数 = 地图大小 * 这个因子
    
    # 训练参数
    BATCH_SIZE = 32
    MEMORY_SIZE = 50000
    LEARNING_RATE = 0.0005
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    
    # 模型参数
    CONV_FILTERS = [16, 32, 32]
    FC_UNITS = 128
    
    # 路径配置
    MAPS_DIR = "maps"
    MODELS_DIR = "models"
    RESULTS_DIR = "results" 