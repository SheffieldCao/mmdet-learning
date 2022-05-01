import os
from glob import glob
from tqdm import tqdm

def check_val_image(image_prefix, gt_prefix):
    img_list = os.listdir(image_prefix)
    full_num = 0
    for img_name in tqdm(img_list):
        city, seq_id, frame_id, _ = img_name.split('.')[0].split('_')
        img_path = os.path.join(image_prefix, img_name)
        assert os.path.isfile(img_path)

        gt_paths = glob(os.path.join(gt_prefix, '*{0}_{1}_{2}_depth.png'.format(city, seq_id, frame_id)))

        assert len(gt_paths) <= 1
        if os.path.isfile(gt_paths[0]):
            print(gt_paths)
            full_num += 1
        
    print('D+I full num: {0}'.format(full_num))

if __name__ == '__main__':
    check_val_image('cityscapes/val', 'dvps_ann/val')
