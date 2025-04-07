import numpy as np
from PIL import Image
import os
import glob

def load_map_from_image(image_path, target_size=(160, 160)):
    """从PNG图像加载SLAM地图数据，并调整大小到指定尺寸
    
    Args:
        image_path: 地图图像路径
        target_size: 目标尺寸，默认(160, 160)
        
    Returns:
        map_data: 地图数据数组，0-可到达，1-障碍物，2-未知区域
    """
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
        
        print(f"从图像加载地图: {image_path}")
        print(f"原始形状: {img.size}, 调整后: {map_data.shape}")
        print(f"可到达区域: {np.sum(map_data == 0)} 像素")
        print(f"不可到达区域: {np.sum(map_data == 1)} 像素")
        print(f"未知区域: {np.sum(map_data == 2)} 像素")
        
        return map_data
    except Exception as e:
        print(f"加载地图图像失败: {e}")
        return None

def get_available_maps(maps_dir="maps"):
    """获取maps目录下所有可用的PNG地图图像
    
    Args:
        maps_dir: 地图目录路径
        
    Returns:
        maps: 地图文件路径列表
    """
    if not os.path.exists(maps_dir):
        print(f"地图目录 {maps_dir} 不存在")
        return []
    return glob.glob(os.path.join(maps_dir, "*.png")) 