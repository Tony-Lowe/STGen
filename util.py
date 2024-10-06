import datetime
import os
import random
import re
from typing import List, Tuple, Union
import cv2
import torch
import numpy as np

import torch.nn as nn
from sklearn.decomposition import PCA
from PIL import Image
import math
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from t3_dataset import insert_spaces
from cldm.ctrl import AttentionStore

def draw_glyph_sp(font,text,rect, scale=1, width=512, height=512, add_space=True):
    cx,cy = rect[0]
    center = (cx*scale,cy*scale)
    w, h = rect[1]
    w*=scale
    h*=scale
    angle = -rect[2]
    img = np.zeros((height*scale, width*scale, 3), np.uint8)
    img = Image.fromarray(img)
    # infer font size
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    # print(_tw,_th)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        # add space
        if len(text) > 1 and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i - 1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.75
        font_size = min(w, h) / (text_w / max(w, h)) * shrink

    new_font = font.font_variant(size=int(font_size))
    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top
    if text_width > img.size[0]:
        shrink = 0.8
        font_size = font_size * img.size[0] / text_width * shrink
        new_font = font.font_variant(size=int(font_size))
        left, top, right, bottom = new_font.getbbox(text)
        text_width = right - left
        text_height = bottom - top
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text(
        (img.size[0] // 2 - text_width // 2, img.size[1] // 2  - text_height // 2 - top),
        text,
        font=new_font,
        fill=(255, 255, 255, 255),
    )
    rotated_layer = layer.rotate(angle, expand=1)
    T_t = np.array([[1, 0, img.size[0]/2-center[0]], [0, 1, img.size[1]/2-center[1]], [0, 0, 1]])
    a,b,c = T_t[0]
    d,e,f = T_t[1]
    rotated_layer = rotated_layer.transform(rotated_layer.size,Image.AFFINE, (a,b,c,d,e,f))

    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    img = np.expand_dims(np.array(img.convert('1')), axis=2).astype(np.float64)
    return img


def draw_pos(ploygon, prob=1.0):
    img = np.zeros((512, 512, 1))
    if random.random() < prob:
        pts = ploygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    return img / 255.0


def sep_mask(pos_imgs, sort_radio="↕"):
    # print(pos_imgs.max())
    if pos_imgs.shape[-1] == 3:
        pos_imgs = cv2.cvtColor(pos_imgs, cv2.COLOR_BGR2GRAY)
    _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image',pos_imgs)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(pos_imgs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    if sort_radio == "↕":
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    else:
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    # poly = []
    # for contour in contours:
    #     rect = cv2.minAreaRect(contour)
    #     box = cv2.boxPoints(rect)
    #     box = np.intp(box)
    #     poly.append(box)
    # print(poly)
    # return poly
    return contours


def count_lines(prompt):
    prompt = prompt.replace("“", '"')
    prompt = prompt.replace("”", '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [" "]
    return len(strs)


def get_lines(prompt):
    prompt = prompt.replace("“", '"')
    prompt = prompt.replace("”", '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [" "]
    for strs_idx, strs_item in enumerate(strs):
        prompt = prompt.replace('"' + strs_item + '"', '"' + f" {PLACE_HOLDER} " + '"')
    return strs, prompt


def check_overlap_polygon(rect_pts1, rect_pts2):
    poly1 = cv2.convexHull(rect_pts1)
    poly2 = cv2.convexHull(rect_pts2)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if (
        rect1[0] + rect1[2] >= rect2[0]
        and rect2[0] + rect2[2] >= rect1[0]
        and rect1[1] + rect1[3] >= rect2[1]
        and rect2[1] + rect2[3] >= rect1[1]
    ):
        return True
    return False


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


smth_3 = GaussianSmoothing(channels=3, sigma=3.0).cuda()

sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).cuda()

sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).cuda()

sobel_x = sobel_x.view(1, 1, 3, 3).expand(1, 3, 3, 3)
sobel_y = sobel_y.view(1, 1, 3, 3).expand(1, 3, 3, 3)

sobel_conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
sobel_conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)


sobel_conv_x.weight = nn.Parameter(sobel_x)
sobel_conv_y.weight = nn.Parameter(sobel_y)


def get_edge(attn_map):
    attn_map_clone = attn_map
    attn_map_clone = attn_map_clone / attn_map_clone.max().detach()
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode="reflect")
    attn_map_clone = smth_3(attn_map_clone)

    sobel_output_x = sobel_conv_x(attn_map_clone).squeeze()
    sobel_output_y = sobel_conv_y(attn_map_clone).squeeze()
    sobel_sum = torch.sqrt(sobel_output_y**2 + sobel_output_x**2)
    return sobel_sum


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


def pca_compute(attn_map, img_size=512,n_c=3):
    attn_map = attn_map.unsqueeze(0)
    # print(attn_map.shape)
    attn_map = nn.Upsample(size=(img_size, img_size), mode="bilinear")(attn_map)
    attn_map = attn_map.squeeze(0).permute(1, 2, 0)
    # print("after upsample", attn_map.shape)
    attn_map = attn_map.reshape(-1, attn_map.shape[-1]).cpu().numpy()
    # print("before_pca", attn_map.shape)
    pca = PCA(n_components=n_c)
    pca.fit(attn_map)
    attn_map_pca = pca.transform(attn_map)  # N X 3
    # print("after pca", attn_map_pca.shape)
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

PLACE_HOLDER = "*"

def get_lines(prompt):
    prompt = prompt.replace("“", '"')
    prompt = prompt.replace("”", '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [" "]
    for strs_idx, strs_item in enumerate(strs):
        prompt = prompt.replace('"' + strs_item + '"', '"' + f" {PLACE_HOLDER} " + '"')
    return strs, prompt
