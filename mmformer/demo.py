import logging
import os
import cv2
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import copy
from PIL import Image
import itertools
from PIL import Image
from typing import Any, Dict, List, Set
import torch
from skimage.morphology.binary import binary_dilation
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    # build_detection_test_loader,
    # build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import EventStorage
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger

from mask2former.utils.addcfg import *
from mask2former import (
    FewShotSemSegEvaluator,  ###
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from mask2former.data import (
    FewShotDatasetMapper_stage2,
    FewShotDatasetMapper_stage1,
    # FewShotVideoDatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
### Adrecover_mapper
)

from buildPredictset import build_all_good_predict

_palette = [
    255, 0, 0, 0, 0, 139, 255, 255, 84, 0, 255, 0, 139, 0, 139, 0, 128, 128,
    128, 128, 128, 139, 0, 0, 218, 165, 32, 144, 238, 144, 160, 82, 45, 148, 0,
    211, 255, 0, 255, 30, 144, 255, 255, 218, 185, 85, 107, 47, 255, 140, 0,
    50, 205, 50, 123, 104, 238, 240, 230, 140, 72, 61, 139, 128, 128, 0, 0, 0,
    205, 221, 160, 221, 143, 188, 143, 127, 255, 212, 176, 224, 230, 244, 164,
    96, 250, 128, 114, 70, 130, 180, 0, 128, 0, 173, 255, 47, 255, 105, 180,
    238, 130, 238, 154, 205, 50, 220, 20, 60, 176, 48, 96, 0, 206, 209, 0, 191,
    255, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45,
    45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51,
    52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58,
    58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64,
    64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70,
    71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77,
    77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83,
    83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89,
    90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96,
    96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101,
    102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106,
    107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111,
    112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116,
    117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121,
    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126,
    127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131,
    132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136,
    137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141,
    142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146,
    147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151,
    152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156,
    157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161,
    162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166,
    167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171,
    172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176,
    177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181,
    182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186,
    187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191,
    192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196,
    197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201,
    202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206,
    207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211,
    212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216,
    217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221,
    222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226,
    227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231,
    232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236,
    237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241,
    242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246,
    247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251,
    252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255, 0, 0, 0
]
color_palette = np.array(_palette).reshape(-1, 3)

def register_adrecover_datasets():
    json_file = './data/adrecover/adrecover.json'
    img_path = './data/adrecover/instance_train'
    register_coco_instances("adrecover0_1shot", {}, json_file, img_path)

def setup(args):
    
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) # ['SEED', k]
    # 
    cfg.merge_from_list(add_seed(cfg))
    # OUTPUT_DIR = os.path.join('out', cfg.DATASETS.dataname, cfg.MODEL.META_ARCHITECTURE, str(cfg.DATASETS.SPLIT))
    DATASETS_TRAIN = (cfg.DATASETS.TRAIN[0] + str(cfg.DATASETS.SPLIT), )
    DATASETS_TEST = (cfg.DATASETS.TEST[0] + str(cfg.DATASETS.SPLIT) +'_'+ str(cfg.DATASETS.SHOT) + 'shot',)
    # 
    print( 'DATASETS.TRAIN', DATASETS_TRAIN, 'DATASETS.TEST', DATASETS_TEST)
    cfg.merge_from_list(['DATASETS.TRAIN', DATASETS_TRAIN, 'DATASETS.TEST', DATASETS_TEST]) # ['SEED', k]
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def overlay(image, mask, colors=[255, 0, 0], cscale=1, alpha=0.4):
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask

        foreground = image * alpha + np.ones(
            image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


def main(args):

    ##register_adrecover_datasets()
    cfg = setup(args)

    model = build_model(cfg)
    new_params = model.state_dict()

    if cfg.MODEL.WEIGHTS_ is not None:
        saved_state_dict = torch.load(cfg.MODEL.WEIGHTS_)['model']
        new_params = model.state_dict()

        for i in saved_state_dict:
            if i in new_params.keys():
                print('\t' + i)
                new_params[i] = saved_state_dict[i]

        model.load_state_dict(new_params)

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        
    model.eval()


    # data_loaders, evaluators = [], []
    # for dataset_name in cfg.DATASETS.TEST:
    #     if dataset_name == "adrecover":
    #         mapper = adrecover_mapper(cfg, False)
    #     elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem":
    #         mapper = FewShotDatasetMapper_stage2(cfg, False)
    #     elif cfg.INPUT.DATASET_MAPPER_NAME == "fewshot_sem_ori":
    #         mapper = FewShotDatasetMapper_stage1(cfg, False)
    #     else:
    #         mapper = None
            
    #      ##mapper = Adrecover_mapper(cfg, False)
    #     data_loader = build_detection_test_loader(cfg, dataset_name, mapper = mapper)
        
    #     data_loaders.append(data_loader)
        
    # count = 0
    # for data in data_loader:
    #     with torch.no_grad():
    #         count = count + 1
    #         print(data)
    #         # print(f"Masking img{data[0]['file_name']}")
    #         # print(data, data[0]['image'].size())
    #         # img = np.array(Image.open(data[0]['file_name']))
    #         # output = model(data)
    #         # output = output['few_shot_result'][0].unsqueeze(0).cuda(non_blocking=True)
    #         # output = F.interpolate(output, size=(data[0]['height'], data[0]['width']), mode='bilinear', align_corners=True)
    #         # output = output.max(1)[1].squeeze(0).cpu().numpy()
    #         # label_index = np.where(output != 0)
    #         # img[label_index] = 0
    #         # out_path = os.path.join("out", "test", f"{count}.png")
    #         # img = Image.fromarray(img)
    #         # img.save(out_path)
    #         if count == 1:                
    #             break


    _root = "./data/adrecover/instance_train"

    count = 0
    num_classes = 2
    
    begin_frame = 1
    end_frame = 60
    for img_p in os.listdir(_root):
        count = count + 1 
        if count < begin_frame: 
            continue
            
        img_path = os.path.join(_root, img_p)
        print(f"Masking img{img_path}")
        predict_dataset = build_all_good_predict(img_path, num_classes, cfg)
        res = np.zeros((predict_dataset[0]['height'], predict_dataset[0]['width'])).astype(np.uint8)
        img = np.array(Image.open(predict_dataset[0]['file_name']), dtype = np.uint8)
        
        if count == begin_frame:
            videoWriter = cv2.VideoWriter("out/test.avi", cv2.VideoWriter_fourcc(*'XVID'), 15,
                                         (predict_dataset[0]['width'], predict_dataset[0]['height']), True)

        for data in predict_dataset:
            # print(data)
            output = model([data])
            output = output['few_shot_result'][0].unsqueeze(0).cuda(non_blocking=True)
            output = F.interpolate(output, (predict_dataset[0]['height'], predict_dataset[0]['width']), 
                                  mode='bilinear', align_corners=True)
            output = output.max(1)[1].squeeze(0).cpu().numpy()
            label_index = np.where(output != 0)
            res[label_index] = 1
        
        outpath = os.path.join("./out/test", img_p)
        overlayed_image = overlay(img, res, color_palette)
        
        videoWriter.write(overlayed_image[..., [2, 1, 0]])
        overlayed_image = Image.fromarray(overlayed_image)
        overlayed_image.save(outpath)
        
        
        # Image.fromarray(res).save(outpath)
        # res = Image.fromarray(res).convert('P')
        # res.putpalette(_palette)
        
        
        # label_index = np.where(res != 1)
        # img[label_index] = 0
        # img = Image.fromarray(img)
        # img.save(outpath)
        # img = cv2.imread(outpath)
        # videoWriter.write(img)

        if count == end_frame:
            videoWriter.release()
            break        

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)