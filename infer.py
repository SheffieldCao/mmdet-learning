from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import cv2
import os
import os.path as osp
from glob import glob
from numpy import random



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
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='cityscapes/demo_{2}/{0}_{1}.jpg'.format('result', name, model_cfg))

if __name__ == '__main__':
    # fetch_val_images()

    # Specify the path to model config and checkpoint file
    config_file = 'outputs/yolact_r50_8x1_cs_2048/yolact_r50_8x1_cs.py'
    checkpoint_file = 'outputs/yolact_r50_8x1_cs_2048/epoch_25.pth'
    model_cfg = '_'.join(checkpoint_file.split('/')[-2:]).replace('.pth', '')
    if not osp.exists(osp.join('cityscapes','demo_{}'.format(model_cfg))):
        os.mkdir(osp.join('cityscapes','demo_{}'.format(model_cfg)))
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:1')
    for img in tqdm(glob('cityscapes/demo/*.png')):
        infer_single_image(model, img, model_cfg)