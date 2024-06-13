from ultralytics import YOLO
import cv2
from PIL import Image
import os
import scipy
import tqdm
import numpy as np
import matplotlib.pyplot as plt


def margen(masks): # reuslts[0]

    conf = masks.boxes.conf     # mask的置信度
    mask = masks.masks[0]       # 用来获得原始图片的shape
    masks = masks.masks         # 所有mask
    final_mask = []
    for _, i in enumerate(masks):
        # 取mask
        mask = i.data[0].cpu().numpy()
        mask = mask.astype(np.uint8)
        x = cv2.Sobel(mask, cv2.CV_16S, 1, 0, ksize=3)
        y = cv2.Sobel(mask, cv2.CV_16S, 0, 1, ksize=3)
        Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
        Scale_absY = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        result = result >= 1
        # 将mask cat到final_mask 第一个维度上
        result = result * float(conf[_])
        final_mask.append(result)

    final_mask = np.array(final_mask)
    final_mask = np.amax(final_mask, axis=0)
    final_mask[0:4,:] = 0
    final_mask[-4:,:] = 0
    final_mask[:,0:4] = 0
    final_mask[:,-4:] = 0
    condition = (final_mask < 0.05) & (final_mask > 0.001)
    final_mask[condition] = 0.05
    return final_mask

def main():
    image_path = "../data/BSDS500/" # 数据集地址
    images = os.listdir(image_path)
    model = YOLO('../finally_100_epoch.pt')  # 模型地址

    for image_name in tqdm.tqdm(images):
        results = model(image_path+image_name, device='0', retina_masks=True, iou=0.7, conf=0.001,imgsz=960,max_det=1000)
        mask_path = "../data/mask/" 
        mat_path = "../data/seg/"       # 保存mat文件的地址
        masks = margen(results[0])

        # 保存成mat文件
        scipy.io.savemat(mat_path + image_name[:-4] + '.mat', {'segs': masks})
        masks = masks.astype(np.uint8)
        masks = masks * 255
        # 保存成图片
        cv2.imwrite(mask_path + image_name[:-4]+".png", masks)

if __name__ == '__main__':
    main()