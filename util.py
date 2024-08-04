import datetime
import os
from typing import List, Tuple, Union
import cv2
import torch
import numpy as np

import torch.nn as nn
from sklearn.decomposition import PCA
from PIL import Image
import math

from cldm.ctrl import AttentionStore


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


def aggregate_attention(
    attention_store: AttentionStore, res: int, from_where: List[int]
):
    out = []
    attention_maps = attention_store.get_average_attention()
    # print(attention_maps.keys())
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[location]:
            if item.shape[1] == num_pixels:
                # torch.Size([8, 256, 77])
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out


def show_ca(prompt: str, attention_maps: torch.Tensor, tokenizer, res: int = 16,select: int=0):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[select, :, :, i]
        max_attention = image.max()
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        image = torch.nn.functional.interpolate(image, size=res**2, mode="bilinear")
        image = image.cpu()
        image = (image - image.min()) / (image.max() - image.min())
        image = image.reshape((res**2, res**2))
        image = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
        image = np.float32(image) / 255
        image = image / np.max(image)
        image = np.uint8(255 * image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res**2, res**2)))
        if i == 0 or i == len(tokens) - 1:
            image = text_under_image(image, decoder(int(tokens[i])))
        else:
            image = text_n_num_under_image(
                image, decoder(int(tokens[i])), max_attention
            )
        images.append(image)

    return view_images(np.stack(images, axis=0))


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + 2 * offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def text_n_num_under_image(
    image: np.ndarray,
    text: str,
    max_val: float,
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + 2 * offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    textsize = cv2.getTextSize(f"Max: {max_val:.4f}", font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + 2 * (offset - textsize[1] // 2)
    cv2.putText(img, f"Max: {max_val:.4f}", (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(
    images: Union[np.ndarray, List],
    num_rows: int = 1,
    offset_ratio: float = 0.02,
    display_image: bool = False,
) -> Image.Image:
    """Displays a list of images in a grid."""
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img
