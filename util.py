import datetime
import os
import cv2
import torch
import numpy as np

import torch.nn as nn
from sklearn.decomposition import PCA
from PIL import Image
import math


def save_images(img_list, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    folder_path = os.path.join(folder, date_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    time_str = now.strftime("%H_%M_%S")
    for idx, img in enumerate(img_list):
        image_number = idx + 1
        filename = f"{time_str}_{image_number}.png"
        save_path = os.path.join(folder_path, filename)
        cv2.imwrite(save_path, img[..., ::-1])


def check_channels(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels > 3:
        image = image[:, :, :3]
    return image


def resize_image(img, max_length=768):
    height, width = img.shape[:2]
    max_dimension = max(height, width)

    if max_dimension > max_length:
        scale_factor = max_length / max_dimension
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    return img


def arr2tensor(arr, bs):
    arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().cuda()
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr


def pca_compute(attn_map, img_size=512):
    attn_map = attn_map.permute(2, 0, 1).unsqueeze(0)
    attn_map = nn.Upsample(size=(img_size, img_size), mode="bilinear")(attn_map)
    attn_map = attn_map.squeeze(0).permute(1, 2, 0)
    attn_map = attn_map.reshape(-1, attn_map.shape[-1]).cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(attn_map)
    attn_map_pca = pca.transform(attn_map)  # N X 3
    h = w = int(math.sqrt(attn_map_pca.shape[0]))
    attn_map_pca = attn_map_pca.reshape(h, w, -1)
    attn_map_pca = (attn_map_pca - attn_map_pca.min(axis=(0, 1))) / (
        attn_map_pca.max(axis=(0, 1)) - attn_map_pca.min(axis=(0, 1))
    )
    pca_img = Image.fromarray((attn_map_pca * 255).astype(np.uint8))
    return pca_img
