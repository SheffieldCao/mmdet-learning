import os
import cv2
from glob import glob
from tqdm import tqdm

img_path = "/mnt/sdf/caoxu/mmdet/cityscapes/video_demo_darmstadt"
depth_path = "/mnt/sdf/caoxu/mmdet/cityscapes/video_demo_results/depth"
ins_path = "/mnt/sdf/caoxu/mmdet/cityscapes/video_demo_results/ins"


videoWriter = cv2.VideoWriter('/mnt/sdf/caoxu/mmdet/cityscapes/depth3.avi', cv2.VideoWriter_fourcc('I','4','2','0'), 30, (2048,1024))
img_paths = glob(os.path.join(depth_path, '*'))
print(len(img_paths))
img_paths.sort()
for path in tqdm(img_paths[:1200]):
    if os.path.exists(path)==False:
        print(path)
    img = cv2.imread(path)
    assert img.shape == (1024,2048,3)
    videoWriter.write(img)
videoWriter.release()
