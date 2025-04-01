import numpy as np
import random

class CoverageEnv:
    def __init__(self, map_data=None, width=10, height=10):
        if map_data is not None:
            self.map_data = map_data
            self.height, self.width = map_data.shape
            self.obstacles = np.where(map_data == 1, True, False)
        else:
            self.width = width
            self.height = height
            self.obstacles = np.zeros((height, width), dtype=bool)
        
        self.observation_space = (self.height, self.width, 3)
        self.action_space = 3  # 前进、左转、右转
        self.reset()

    # ... 其余代码保持不变 ... 