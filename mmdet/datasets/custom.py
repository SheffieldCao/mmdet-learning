# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from PIL import Image, ImageOps

from mmcv.utils import print_log
from terminaltables import AsciiTable
import torch
from torch.utils import data
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose

class BaseDataset(data.Dataset):
    """
    Base class for segmentation dataset.
    Arguments:
        root: Str, root directory.
        split: Str, data split, e.g. train/val/test.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
    """
    def __init__(self,
                 root,
                 split,
                 is_train=True,
                 crop_size=(513, 1025),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 **kwargs):
        self.root = root
        self.split = split
        self.is_train = is_train

        # pre_scale
        self.pre_scale = None
        if kwargs.get('has_pre_scale', False):
            self.pre_scale = kwargs['pre_scale']

        self.crop_h, self.crop_w = crop_size

        self.mirror = mirror
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step_size = scale_step_size

        self.mean = mean
        self.std = std

        self.pad_value = tuple([int(v * 255) for v in self.mean])

        # ======== override the following fields ========
        self.ignore_label = 255
        self.label_pad_value = (self.ignore_label, )
        self.label_dtype = 'uint8'

        # list of image filename (required)
        self.img_list = []
        # list of label filename (required)
        self.ann_list = []
        # list of instance dictionary (optional)
        self.ins_list = []

        self.has_instance = False
        self.label_divisor = 1000

        self.raw_label_transform = None
        self.pre_augmentation_transform = None
        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # TODO: handle transform properly when there is no label
        dataset_dict = {}
        assert osp.exists(self.img_list[index]), 'Path does not exist: {}'.format(self.img_list[index])
        image = self.read_image(self.img_list[index], 'RGB', pre_scale = self.pre_scale)
        if not self.is_train:
            # Do not save this during training.
            dataset_dict['raw_image'] = image.copy()
        if self.ann_list is not None:
            assert osp.exists(self.ann_list[index]), 'Path does not exist: {}'.format(self.ann_list[index])
            label = self.read_label(self.ann_list[index], self.label_dtype, pre_scale = self.pre_scale)
        else:
            label = None
        raw_label = label.copy()
        if self.raw_label_transform is not None:
            raw_label = self.raw_label_transform(raw_label, self.ins_list[index])['semantic']
        if not self.is_train:
            # Do not save this during training
            dataset_dict['raw_label'] = raw_label
        size = image.shape
        dataset_dict['raw_size'] = np.array(size)
        # To save prediction for official evaluation.
        name = osp.splitext(osp.basename(self.ann_list[index]))[0]
        # TODO: how to return the filename?
        # dataset_dict['name'] = np.array(name)

        # Resize and pad image to the same size before data augmentation.
        if self.pre_augmentation_transform is not None:
            image, label = self.pre_augmentation_transform(image, label)
            size = image.shape
            dataset_dict['size'] = np.array(size)
        else:
            dataset_dict['size'] = dataset_dict['raw_size']

        # Apply data augmentation.
        if self.transform is not None:
            image, label = self.transform(image, label)

        dataset_dict['image'] = image
        if not self.has_instance:
            dataset_dict['semantic'] = torch.as_tensor(label.astype('long'))
            return dataset_dict

        # Generate training target.
        if self.target_transform is not None:
            label_dict = self.target_transform(label, self.ins_list[index])
            for key in label_dict.keys():
                dataset_dict[key] = label_dict[key]
        return dataset_dict

    @staticmethod
    def read_image(file_name, format=None, **kwargs):
        if kwargs['pre_scale'] is not None:
            image = Image.open(file_name).resize(kwargs['pre_scale'], Image.LANCZOS)
        else:
            image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

    @staticmethod
    def read_label(file_name, dtype='uint8', **kwargs):
        # In some cases, `uint8` is not enough for label
        if  kwargs['pre_scale'] is not None:
            label = Image.open(file_name).resize(kwargs['pre_scale'], Image.LANCZOS)
        else:
            label = Image.open(file_name)
        return np.asarray(label, dtype=dtype)

    def reverse_transform(self, image_tensor):
        """Reverse the normalization on image.
        Args:
            image_tensor: torch.Tensor, the normalized image tensor.
        Returns:
            image: numpy.array, the original image before normalization.
        """
        dtype = image_tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=image_tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=image_tensor.device)
        image_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        image = image_tensor.mul(255)\
                            .clamp(0, 255)\
                            .byte()\
                            .permute(1, 2, 0)\
                            .cpu().numpy()
        return image

    @staticmethod
    def train_id_to_eval_id():
        return None

@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 do_kitti_benchmark_crop=False,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.do_kitti_benchmark_crop = do_kitti_benchmark_crop
        self.file_client = mmcv.FileClient(**file_client_args)
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            if hasattr(self.file_client, 'get_local_path'):
                with self.file_client.get_local_path(
                        self.proposal_file) as local_path:
                    self.proposals = self.load_proposals(local_path)
            else:
                warnings.warn(
                    'The used MMCV version does not have get_local_path. '
                    f'We treat the {self.ann_file} as local paths and it '
                    'might cause errors if the path is not a local path. '
                    'Please use MMCV>= 1.3.16 if you meet errors.')
                self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

        table = AsciiTable(table_data)
        result += table.table
        return result
