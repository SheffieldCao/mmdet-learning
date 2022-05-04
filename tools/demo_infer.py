import argparse
from tqdm import tqdm
import cv2
import os
import os.path as osp
from glob import glob
import numpy as np
from numpy import random

from mmdet.apis import init_detector, inference_detector


DATASET_PREFIX = '/mnt/sdf/caoxu/'

def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config_file', help='eval config file path')
    parser.add_argument('checkpoint_file', help='eval config file path')
    parser.add_argument('task', type=str, default='demo', help="task to do ['test', 'demo']")
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

toLabelID = {
    '0': '24',
    '1': '25',
    '2': '26',
    '3': '27',
    '4': '28',
    '5': '31',
    '6': '32',
    '7': '33',
}

def save_single_result(result, name, test_save_prefix):
    single_txt_name = name + '_TestSet_InstancePredInfo.txt'
    single_mask_name = name + '_TestSet_InstanceMask_'
    single_txt_save_path = os.path.join(test_save_prefix, single_txt_name)
    assert len(result) == 2
    bboxs,masks = result
    assert len(bboxs) == len(masks) and len(bboxs) == 8

    f = open(single_txt_save_path, 'w')
    for i,(bbox,mask) in enumerate(zip(bboxs,masks)):
        assert bbox.shape[0] == len(mask), "bbox num in a single image doesn't match mask num"
        assert bbox.shape[1] == 5, "bbox dim=1 shape error"
        for j in range(bbox.shape[0]):
            # box_params = bbox[j,:]
            labelIDx = int(toLabelID[str(int(i))])
            box_coff = bbox[j,-1]
            
            # save mask
            single_mask_namex = single_mask_name + 'labelID{}_Index{}.png'.format(labelIDx, j)
            cv2.imwrite(os.path.join(test_save_prefix, single_mask_namex), 255*mask[j].astype(np.int))

            # write to txt
            single_mask_pathx = './'+single_mask_namex

            f.write(single_mask_pathx+' ')
            f.write(str(labelIDx)+' ')
            f.write(str(box_coff))
            f.write('\n')
    
    f.close()

def infer_single_image(model, img, model_cfg):

    name, _ = img.split('/')[-1].split('.')
    # test a single image and show the results
    result = inference_detector(model, img)
    # save the visualization results to image files
    model.show_result(img, result, out_file=DATASET_PREFIX+'datasets/demo_results/demo_{2}/{0}_{1}.jpg'.format('result', name, model_cfg))

def gen_test_results(config, checkpoint, gpu_id=0, test_image_prefix="cityscapes/leftImg8bit/test", test_save_prefix = "cityscapes/test_results"):
    from glob import glob
    if not os.path.exists(test_save_prefix):
        os.mkdir(test_save_prefix)
    img_list = glob(osp.join(test_image_prefix, '*/*.png'))

    model = init_detector(config, checkpoint, device='cuda:{}'.format(gpu_id))
    for img in tqdm(img_list):
        name, _ = img.split('/')[-1].split('.')
        name = '_'.join(name.split('_')[:-1])
        # test a single image and show the results
        result = inference_detector(model, img)

        save_single_result(result, name, test_save_prefix)

def main():
    args = parse_args()
    # args.config_file = 'outputs/mask_rcnn_x50_32x4d_dw_gn_cs_8x2_cs_1024/mask_rcnn_x50_32x4d_dw_gn_cs.py'
    # args.checkpoint_file = 'outputs/mask_rcnn_x50_32x4d_dw_gn_cs_8x2_cs_1024/epoch_3.pth'
    if args.task == 'demo': 
        model_cfg = '_'.join(args.checkpoint_file.split('/')[-2:]).replace('.pth', '')
        if not osp.exists(DATASET_PREFIX+'datasets/demo_results/demo_{}'.format(model_cfg)):
            os.mkdir(DATASET_PREFIX+'datasets/demo_results/demo_{}'.format(model_cfg))
        # if not osp.exists(osp.join('cityscapes','demo_{}'.format(model_cfg))):
        #     os.mkdir(osp.join('cityscapes','demo_{}'.format(model_cfg)))
        # build the model from a config file and a checkpoint file
        model = init_detector(args.config_file, args.checkpoint_file, device='cuda:{}'.format(args.gpu_id))
        for img in tqdm(glob('cityscapes/demo/*.png')):
            infer_single_image(model, img, model_cfg)
    elif args.task == 'test':
        gen_test_results(args.config_file, args.checkpoint_file, args.gpu_id)
    else:
        raise ValueError('Unknown task!')


if __name__ == '__main__':
    # fetch_val_images()
    
    # demo inference
    main()

    # generate test set results
    # gen_test_results('outputs/mask_rcnn_x50_32x4d_dw_gn_cs_8x2_cs_1024/mask_rcnn_x50_32x4d_dw_gn_cs.py', 'outputs/mask_rcnn_x50_32x4d_dw_gn_cs_8x2_cs_1024/epoch_33.pth', 7)