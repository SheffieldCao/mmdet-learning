import argparse
from tqdm import tqdm
import cv2
import os
import os.path as osp
from glob import glob
from numpy import random

from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config_file', help='eval config file path')
    parser.add_argument('checkpoint_file', help='eval config file path')
    parser.add_argument('--gpu_id', type=int, default=0, help='random seed')

    args = parser.parse_args()

    return args

def fetch_val_images():
    val_set = 'cityscapes/val'
    demo_val_prefix = 'cityscapes/demo'
    img_paths = glob(osp.join(val_set, '*.png'))
    idx = random.choice(len(img_paths),10)
    selected_img_paths = [img_paths[i] for i in idx]
    for i,j in enumerate(selected_img_paths):
        img = cv2.imread(j)
        cv2.imwrite(osp.join(demo_val_prefix, '{}.png'.format(j.split('/')[-1].split('.')[0])), img)

def infer_single_image(model, img, model_cfg):

    name, _ = img.split('/')[-1].split('.')
    # test a single image and show the results
    result = inference_detector(model, img)
    # save the visualization results to image files
    model.show_result(img, result, out_file='~/datasets/demo_{2}/{0}_{1}.jpg'.format('result', name, model_cfg))

def main():
    args = parse_args()
    # args.config_file = 'outputs/mask_rcnn_x50_32x4d_dw_gn_cs_8x2_cs_1024/mask_rcnn_x50_32x4d_dw_gn_cs.py'
    # args.checkpoint_file = 'outputs/mask_rcnn_x50_32x4d_dw_gn_cs_8x2_cs_1024/epoch_3.pth'

    model_cfg = '_'.join(args.checkpoint_file.split('/')[-2:]).replace('.pth', '')
    if not osp.exists(osp.join('cityscapes','demo_{}'.format(model_cfg))):
        os.mkdir(osp.join('cityscapes','demo_{}'.format(model_cfg)))
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config_file, args.checkpoint_file, device='cuda:1')
    for img in tqdm(glob('cityscapes/demo/*.png')):
        infer_single_image(model, img, model_cfg)

if __name__ == '__main__':
    # fetch_val_images()
    main()