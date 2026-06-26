import cv2
import numpy as np
import os
import time

def add_heavy_noise_and_blur(image_path, output_path):
    """人为制造极端恶劣环境：模拟极度暗光下的传感器高斯噪点与剧烈运动模糊"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 1. 添加严重的高斯噪声 (模拟高ISO底噪)
    mean = 0
    sigma = 1 # 噪声强度极大
    gauss = np.random.normal(mean, sigma, img.shape).astype('float32')
    noisy_img = np.clip(img.astype('float32') + gauss, 0, 255).astype('uint8')
    
    # 2. 添加严重的运动模糊 (模拟严重的手抖或设备高速运动)
    size = 27
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    blurred_img = cv2.filter2D(noisy_img, -1, kernel)
    
    cv2.imwrite(output_path, blurred_img)

if __name__ == "__main__":
    input_dir = "./test/my_test_inputs"
    robust_dir = "./test/my_test_robustness"
    
    # 创建文件夹
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(robust_dir, exist_ok=True)
    
    # 检查是否有输入图片
    valid_images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(valid_images) == 0:
        print(f"请先在 {input_dir} 文件夹中放入几张夜景测试图片！")
        exit()
    
    print("====== 步骤 1: 正在生成极端环境压力测试数据集 ======")
    for img_name in valid_images:
        add_heavy_noise_and_blur(
            os.path.join(input_dir, img_name),
            os.path.join(robust_dir, img_name)
        )
    print(f"退化测试图创建完毕，已存入: {robust_dir}")
    
    print("\n====== 步骤 2: 启动 DarkIR 极端鲁棒性推理 ======")
    start_time = time.time()
    
    # 直接调用你刚才已经修好显存爆掉问题的 inference.py
    os.system(f"CUDA_VISIBLE_DEVICES=0 python inference.py -i {robust_dir}")
    
    end_time = time.time()
    print(f"====== 测试完成！总推理耗时: {end_time - start_time:.4f} 秒 ======")
    print("请前往 ./images/results 文件夹查看修复结果！")