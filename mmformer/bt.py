import torch

total_list = []

for i in range(1, 71):
    img_path = f'instance_val/{str(i).zfill(6)}.jpg'
    mask_path = f'annotations/coco_masks/instance_train/{str(i).zfill(6)}.png'
    img_mask_pair = (img_path, mask_path)
    total_list.append(img_mask_pair)

for i in range(1, 2):
    img_path = f'support_image/image{i}.jpg'
    mask_path = f'support_mask/mask{i}.png'
    img_mask_pair = (img_path, mask_path)
    total_list.append(img_mask_pair)

content = (total_list, {1: total_list, 2:total_list})

torch.save(content, "list/adrecover/adrecover_split3.pth")
