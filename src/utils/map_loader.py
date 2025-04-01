from PIL import Image
import numpy as np

def load_map_from_image(image_path, target_size=(32, 32)):
    """从PNG图像加载地图数据"""
    try:
        img = Image.open(image_path)
        img = img.resize(target_size, Image.BILINEAR)
        if img.mode != 'L':
            img = img.convert('L')
        map_data = np.array(img)
        map_data = np.where(map_data < 128, 1, 0)
        
        print(f"地图加载成功:")
        print(f"- 文件: {image_path}")
        print(f"- 尺寸: {map_data.shape}")
        print(f"- 障碍物比例: {np.sum(map_data) / (map_data.shape[0] * map_data.shape[1]):.2%}")
        return map_data
    except Exception as e:
        print(f"地图加载失败: {e}")
        return None 