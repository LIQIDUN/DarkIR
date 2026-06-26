import os
from PIL import Image
import cv2 as cv
from options.options import parse
import argparse
from archs.retinexformer import RetinexFormer
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description="Script for prediction")
parser.add_argument('-p', '--config', type=str, default='./options/inference/LOLBlur.yml', help = 'Config file of prediction')
parser.add_argument('-i', '--inp_path', type=str, default='./images/inputs', 
                help="Folder path")
args = parser.parse_args()


path_options = args.config
opt = parse(path_options)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# PyTorch library
import torch
import torch.optim
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision.transforms import Resize

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.test_utils import *
from ptflops import get_model_complexity_info

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

#define some auxiliary functions
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    
    return img
def normalize_tensor(tensor):
    
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    
    tensor = tensor.squeeze(0)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    
    return tensor

def load_model(model, path_weights):
    map_location = 'cpu'
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
   
    weights = checkpoints['params']
    weights = {'module.' + key: value for key, value in weights.items()}

    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    print(macs, params)
    model.load_state_dict(weights)
    print('Loaded weights correctly')
    
    return model

#parameters for saving model
PATH_MODEL = opt['save']['path']
resize = opt['Resize']

def predict_folder(rank, world_size):
    
    setup(rank, world_size=world_size, Master_port='12354')
    
    # DEFINE NETWORK, SCHEDULER AND OPTIMIZER
    model, _, _ = create_model(opt['network'], rank=rank)

    model = load_model(model, path_weights = opt['save']['path'])
    # create data
    PATH_IMAGES= args.inp_path
    PATH_RESULTS = './images/results'

    #create folder if it doen't exist
    not os.path.isdir(PATH_RESULTS) and os.mkdir(PATH_RESULTS)

    path_images = [os.path.join(PATH_IMAGES, path) for path in os.listdir(PATH_IMAGES) if path.endswith(('.png', '.PNG', '.jpg', '.JPEG'))]
    path_images = [file for file in path_images if not file.endswith('.csv') and not file.endswith('.txt')]
   
    model.eval()
    if rank==0:
        pbar = tqdm(total = len(path_images))

    for path_img in path_images:
        tensor = path_to_tensor(path_img).to(device)
        _, _, H, W = tensor.shape
        
        # 适配显存限制
        max_dim = max(H, W)
        # 设定最大安全边长阈值
        if max_dim > 1080:  
            scale = 1080 / max_dim
            new_size = [int(H * scale), int(W * scale)]
            downsample = Resize(new_size)
            do_resize = True
        else:
            downsample = torch.nn.Identity()
            do_resize = False
            
        tensor = downsample(tensor)
        tensor = pad_tensor(tensor) 

        # 抑制过曝光晕
        with torch.no_grad():
            # 长曝光推理：提取暗部细节，但高光会过曝
            out_normal = model(tensor, side_loss=False)
            
            # 短曝光推理：通过线性衰减输入以压缩动态范围
            gamma_ratio = 0.35 # 压暗系数
            tensor_dark = tensor * gamma_ratio 
            out_dark = model(tensor_dark, side_loss=False)
            
        # 若触发降采样，则通过插值恢复至原始空间分辨率
        if do_resize:
            upsample = Resize((H, W))
            out_normal = upsample(out_normal)
            out_dark = upsample(out_dark)
        else: 
            upsample = torch.nn.Identity()

        # 数据格式转换：PyTorch Tensor 转为 OpenCV 兼容格式
        import numpy as np
        out_n_np = (torch.clamp(out_normal[:,:,:H,:W], 0, 1).squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        out_d_np = (torch.clamp(out_dark[:,:,:H,:W], 0, 1).squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        
        out_n_bgr = cv.cvtColor(out_n_np, cv.COLOR_RGB2BGR)
        out_d_bgr = cv.cvtColor(out_d_np, cv.COLOR_RGB2BGR)

        # 多曝光融合：利用 Mertens 算法与拉普拉斯金字塔，进行加权缝合
        merge_mertens = cv.createMergeMertens()
        fusion_float = merge_mertens.process([out_d_bgr, out_n_bgr]) 
        improved_img = np.clip(fusion_float * 255.0, 0, 255).astype(np.uint8)

        # 保存基准结果和改进结果
        cv.imwrite(os.path.join(PATH_RESULTS, "Baseline_" + os.path.basename(path_img)), out_n_bgr)
        cv.imwrite(os.path.join(PATH_RESULTS, "Improved_" + os.path.basename(path_img)), improved_img)
        pbar.update(1)

        pass

    print('Finished inference!')
    if rank == 0:
        pbar.close()   
    cleanup()

def main():
    world_size = 1
    print('Used GPUS:', world_size)
    mp.spawn(predict_folder, args =(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()










