import os
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform

def build_all_good_predict(img_file, class_num, cfg):
    """
    img_file: img_filename
    class_num: num of useful class
    cfg: yaml file, config of project

    return a list of dict, each in it is an input of Detectron2 model input
    with different class_chosen
    """

    _support_root = "./data/adrecover/support/"
    predict_set = []
    ret = from_config(cfg, False)

    for i in range(1, class_num + 1):

        image = utils.read_image(img_file, format=cfg.INPUT.FORMAT)
        w, h = image.shape[1], image.shape[0]
        image, _ = aug(image, image, ret)
        support_image_name = os.path.join(_support_root, f"image_{i}_1.jpg")
        support_mask_name = os.path.join(_support_root, f"mask_{i}_1.png")
        support_image = utils.read_image(support_image_name, format=cfg.INPUT.FORMAT)
        support_mask = utils.read_image(support_mask_name).astype("double")
        support_image, support_mask = aug(support_image, support_mask, ret)
        support_image, support_mask = support_image.unsqueeze(0), support_mask.unsqueeze(0).float()

        class_i_dict = {'file_name':img_file,
                        'width':w,
                        'height':h,
                        'image':image,
                        'support_img':support_image,
                        'support_label':support_mask}
        predict_set.append(class_i_dict)

    return predict_set


def aug(image, sem_seg_gt, ret):
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(ret["augmentations"], aug_input)
        image_ = aug_input.image
        sem_seg_gt_ = aug_input.sem_seg
 
        # Pad image and segmentation label here!
        image_ = torch.as_tensor(np.ascontiguousarray(image_.transpose(2, 0, 1)))
        sem_seg_gt_ = torch.as_tensor(sem_seg_gt_.astype("long"))

        if ret["size_divisibility"] > 0:
            image_size = (image_.shape[-2], image_.shape[-1])
            padding_size = [
                0,
                ret["size_divisibility"] - image_size[1],
                0,
                ret["size_divisibility"] - image_size[0],
            ]
            image_ = F.pad(image_, padding_size, value=128).contiguous()
            if sem_seg_gt_ is not None:
                sem_seg_gt_ = F.pad(
                    sem_seg_gt_, padding_size, value=ret["ignore_label"]
                ).contiguous()
        
        return image_, sem_seg_gt_


def from_config(cfg, is_train=True):
        # Build augmentation
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())

            # Assume always applies to the training set.
            dataset_names = cfg.DATASETS.TRAIN
            augs2 = augs

        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            augs = [T.Resize(min_size)]
            dataset_names = cfg.DATASETS.TEST

        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = 255

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY if is_train else -1,
            "split": cfg.DATASETS.SPLIT,
            "shot": cfg.DATASETS.SHOT,
            "ignore_bg": cfg.DATASETS.IGNORE_BG,
            "dataname": cfg.DATASETS.dataname,
            "MYSIZE": cfg.INPUT.MYSIZE,
            "root": cfg.DATASETS.IMGPATHROOT
        }
        return ret

