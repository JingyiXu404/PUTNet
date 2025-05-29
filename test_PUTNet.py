"""
@File: CatNet_test.py
@Time: 2022/11/6
@Author: rp
@Software: PyCharm

"""
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from PUTNet import PUTNet
from data_y3 import get_loader, test_dataset
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
parser.add_argument('--testset', type=str, default='LFSD', help='train from checkpoints')
parser.add_argument('--load_pre', type=str, default='./cpts/', help='train from checkpoints')
parser.add_argument('--test_path',type=str,default='/raid/xjy/dataset/RGBD_SOD/testset/',help='test dataset path')
parser.add_argument('--save_path',type=str,default='./result/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

model = PUTNet()
device = torch.device('cuda' if opt.gpu_id is not None else 'cpu')

def data_process_npy_img(data_in):
    data_out=data_in.detach().float().cpu().numpy()
    data_out=np.transpose(data_out,(0,2,3,1))
    data_out = data_out.squeeze()
    return data_out*255

def load_pth(model,gpu_id,device,pth_path):
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print('USE GPU- ', gpu_id)
        model = nn.DataParallel(model).to(device)
        if (pth_path is not None):
            if os.path.exists(pth_path):
                model.load_state_dict(torch.load(pth_path))
                print('load model from ', pth_path)
    else:
        model = model.to(device)
        if (pth_path is not None):
            if os.path.exists(pth_path):
                checkpoint = torch.load(pth_path,map_location=torch.device('cpu'))
                new_checkpoint = {}
                for k in checkpoint.keys():
                    if k.startswith('module.'):
                        new_key = k[len('module.'):]
                        new_checkpoint[new_key] = checkpoint[k]
                    else:
                        new_checkpoint[k] = checkpoint[k]
                model.load_state_dict(new_checkpoint)
                print('load model from ', pth_path)
    model.eval()
    return model


# test
def data_process_save(res,file,name):
    n,c,h,w = res.shape
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    for i in range(c):
        cv2.imwrite(file + name.replace('.png', '_' + str(i) + '.png'), res[i]*255)

test_datasets = opt.testset.split(',')
print(test_datasets)
for dataset in test_datasets:
    pth_path = opt.load_pre + dataset + '.pth'
    save_path = opt.save_path + dataset + '/'
    model = load_pth(model,opt.gpu_id,device,pth_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path,exist_ok=True)

    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, image3, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        depth = depth.to(device)
        depth3 = depth.repeat(1,3,1,1).to(device)
        image3 = image3.to(device)
        preds, masks = model(image, image3, depth, depth3)
        res = preds[-1]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)
    print('Test Done!')
