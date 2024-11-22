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
    contours = [cnt for cnt in contours if cnt.shape[0] > 20]
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

def AdaINnorm(x_tar, glyph_latent, rect_mask):
    """
    Norm the value inside the rect_mask
    """
    valid_pixel = rect_mask.sum()
    x_mean = (x_tar * rect_mask).sum(dim=(-2,-1))/valid_pixel
    x_std = ((x_tar * rect_mask).pow(2).sum(dim=(-2,-1)) / valid_pixel - x_mean.pow(2)).sqrt()
    x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
    x_std = x_std.unsqueeze(-1).unsqueeze(-1)
    g_mean = (glyph_latent * rect_mask).sum(dim=(-2, -1)) / valid_pixel
    g_std = ((glyph_latent * rect_mask).pow(2).sum(dim=(-2,-1)) / valid_pixel - g_mean.pow(2)).sqrt()
    g_mean = g_mean.unsqueeze(-1).unsqueeze(-1)
    g_std = g_std.unsqueeze(-1).unsqueeze(-1)
    glyph_latent = (glyph_latent - g_mean) / g_std * x_std + x_mean
    return glyph_latent


def compute_rotate_rect(
    polygon: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    获取多边形旋转参数，其中角度为顺时针长边角度
    Args:
        polygon: 多边形顶点坐标，形状为 (N, 2)
    Returns:
        center: 多边形中心点坐标，形状为 (2,)
        size: 多边形长宽，形状为 (2,)
        angle: 多边形旋转角度，范围为 (0, 180]
    """
    rect = cv2.minAreaRect(polygon)

    center = rect[0]
    width, height = rect[1]
    angle = rect[2]

    if width < height:
        width, height = height, width
        angle += 90

    return center, (width, height), angle


def rotate_points(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """
    旋转点集
    Args:
        points: 点集，形状为 (N, 2)
        center: 旋转中心，形状为 (2,)
        angle: 旋转角度，单位为度
    Returns:
        points: 旋转后的点集，形状为 (N, 2)
    """
    rotate_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    points = points.reshape(-1, 2)
    points = np.concatenate([points, np.ones([len(points), 1])], axis=-1)
    points = (rotate_mat @ points.T).T
    return points


def generate_bezier_cubic(control_points, t):
    """
    生成三次Bezier曲线上的点
    Args:
        control_points: 三次Bezier曲线控制点坐标，形状为 (4, 2)
        t: 参数，范围为 [0, 1]
    Returns:
        x, y: 三次Bezier曲线上的点坐标
    """
    P = np.array(control_points).reshape(-1, 2)
    M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
    T = np.array([[t**3, t**2, t, 1]])
    B = np.matmul(np.matmul(T, M), P)
    return B[0]


def fit_bezier_curve(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    利用三次Bezier曲线拟合多边形
    设多边形顶点points为p，控制点为c，并记B为三次Bezier基函数矩阵。
    则有B@c为拟合坐标，损失函数为||B@c-p||^2
    求导并令导数为0，可得B.T@B@c=B.T@p，利用np.linalg.solve求解c
    Args:
        points: 多边形顶点坐标，形状为 (N, 2)
    Returns:
        bezier_points: 三次Bezier曲线控制点坐标，形状为 (4, 2)
        error: 拟合误差
    """

    # 计算折线长度，并基于折线长度推导顶点权重与时间，W可以被省略
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    W = np.concatenate([lengths[:1], lengths[1:] + lengths[:-1], lengths[-1:]]) / 2
    W = np.diag(W)
    ts = np.concatenate([[0.0], np.cumsum(lengths)]) / np.sum(lengths)

    M = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
    T = np.array([[t**3, t**2, t, 1] for t in ts])
    B = T @ M

    control_points_x = np.linalg.solve(B.T @ W.T @ B, B.T @ W.T @ points[:, 0])
    control_points_y = np.linalg.solve(B.T @ W.T @ B, B.T @ W.T @ points[:, 1])
    control_points = np.array([control_points_x, control_points_y]).T

    # 逐点计算二次误差
    fit_error = np.square(
        np.array([generate_bezier_cubic(control_points, t) for t in ts]) - points
    ).sum()
    return control_points, fit_error


def fit_bezier_curve_pair(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    利用两条三次Bezier曲线拟合多边形
    将多边形划分为上壳与下壳，分别拟合三次Bezier曲线
    尝试将最左侧与最右侧点及其相邻点分别作为上壳与下壳的起点，利用最小二乘法拟合三次Bezier曲线
    Args:
        points: 多边形顶点坐标，形状为 (N, 2)
    Returns:
        bezier_points: 上壳与下壳Bezier曲线控制点坐标，形状均为 (M, 2)
    """
    points = points.reshape(-1, 2).astype(np.float32)
    # 判断points是否逆时针排列，若是则翻转
    if cv2.contourArea(points, oriented=True) < 0:
        points = points[::-1]
    points = points.reshape(-1, 2)
    # 找到最左侧与最右侧点
    left_k = np.argmin(points[:, 0])
    right_k = np.argmax(points[:, 0])
    assert left_k != right_k

    # 尝试将最左侧与最右侧点及其相邻点分别作为上壳与下壳的起点，利用最小二乘法拟合三次Bezier曲线
    fit_curves, fit_errors = [], []
    for left_begin in range(left_k - 1, left_k + 2):
        for right_begin in range(right_k - 1, right_k + 2):

            # 计算实际的起点索引
            left_begin = left_begin % len(points)
            right_begin = right_begin % len(points)
            if left_begin == right_begin:
                continue

            # 根据起点索引确定上壳与下壳的点集
            if left_begin > right_begin:
                up_points = np.concatenate([points[left_begin:], points[:right_begin]])
                down_points = points[right_begin:left_begin]
            else:
                up_points = points[left_begin:right_begin]
                down_points = np.concatenate(
                    [points[right_begin:], points[:left_begin]]
                )

            # 分别拟合三次Bezier曲线
            up_curve, up_error = fit_bezier_curve(up_points)
            down_curve, down_error = fit_bezier_curve(down_points)

            fit_curves.append((up_curve, down_curve))
            fit_errors.append(up_error + down_error)

    # 选择误差最小的拟合结果
    return fit_curves[np.argmin(fit_errors)]


def estimate_text_size(
    font: ImageFont.FreeTypeFont, text: str, width: float, height: float
) -> tuple[str, ImageFont.FreeTypeFont]:
    """
    估计文本框尺寸，并基于文本框尺寸预测文本大小
    Args:
        text: 文本
        width: 文本框宽度
        height: 文本框高度
    Returns:
        text: 文本
        font_size: 文本大小
    """
    init_ratio, step_ratio = 0.8, 0.9
    font_size = height * init_ratio
    (left, top, right, bottom) = font.font_variant(size=round(font_size)).getbbox(text)
    text_width = right - left
    text_height = bottom - top
    if text_width > width:
        font_size *= width / text_width
    else:
        for i in range(1, 100):
            adjust_text = (" " * i).join(text)
            (left, top, right, bottom) = font.font_variant(
                size=round(font_size)
            ).getbbox(adjust_text)
            text_width = right - left
            text_height = bottom - top
            if text_width > width:
                break
        text = (" " * (i - 1)).join(text)
    return text, font.font_variant(size=round(font_size))


def draw_polygon(
    width: int,
    height: int,
    polygon: np.ndarray,
    color: tuple[int, int, int] | str = "white",
) -> Image.Image:
    """
    绘制多边形轮廓
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(p) for p in polygon.reshape(-1, 2)], outline=color, width=2)
    # img.save('./draw_glyph_test/polygon.jpg')
    return img


def draw_glyph_curve(
        font: ImageFont.FreeTypeFont = None, text: str = '', polygon: np.ndarray = None, 
        scale: float = 1, width: int = 512, height: int = 512, 
        add_space: bool = True, offset_angle: float = 0,discard_ratio: float=0.1
        ) -> tuple[np.ndarray, float, tuple]:
    
    enlarge_polygon = polygon * scale

    # 获取多边形旋转中心、长宽与旋转角度
    rect = compute_rotate_rect(enlarge_polygon)
    center = rect[0]
    angle = rect[2]
    # 调整角度
    if angle > 90:
        angle -= 180

    # 旋转多边形至水平，并拟合三次Bezier曲线
    flat_polygon = rotate_points(enlarge_polygon, center, angle)
    bezier_control_points = fit_bezier_curve_pair(flat_polygon)
    up_control_points = rotate_points(bezier_control_points[0], center, -angle)
    down_control_points = rotate_points(bezier_control_points[1], center, -angle)[::-1]
    avg_control_points = (up_control_points + down_control_points) / 2
    del bezier_control_points
    
    up_points = np.array([generate_bezier_cubic(up_control_points, t) for t in np.linspace(0, 1, 100)])
    down_points = np.array([generate_bezier_cubic(down_control_points, t) for t in np.linspace(0, 1, 100)])
    avg_points = (up_points + down_points) / 2

    # img = draw_polygon(width * scale, height * scale, np.concatenate([up_points, down_points[::-1]], axis=0), color='red')
    # img.save('draw_glyph_test/bezier_curve.png')
    # img = draw_polygon(width * scale, width * scale, avg_points, color='red')
    #  img.save('draw_glyph_test/bezier_curve_avg.png')
    
    # 估计文本框尺寸，并基于文本框尺寸预测文本大小，需要丢弃曲线两端一定比例
    discard_ratio = 0.1
    bezier_width = min(np.linalg.norm(avg_points[1:] - avg_points[:-1], axis=1).sum(), 
                       np.linalg.norm(up_points[1:] - up_points[:-1], axis=1).sum(),
                       np.linalg.norm(down_points[1:] - down_points[:-1], axis=1).sum())
    bezier_height = np.linalg.norm(up_points - down_points, axis=1).max()
    text, font = estimate_text_size(font, text, bezier_width * (1 - 2 * discard_ratio), bezier_height)
    
    # 获取逐字符时间
    char_times = [0.]
    for c in text:
        c_left, _, c_right, _ = font.getbbox(c)
        char_times.append(char_times[-1] + c_right - c_left)
    char_times = np.array(char_times)


    text_time = char_times[-1] / bezier_width
    char_times = char_times / char_times[-1] * text_time
    char_times -= (text_time - 1 ) / 2

    img = Image.new('RGBA', (width * scale, height * scale), (0, 0, 0, 0))
    for i, (c, left_time, right_time) in enumerate(zip(text, char_times[:-1], char_times[1:])):
        img_c = Image.new('RGBA', (width * scale, height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img_c)
        center_time = (left_time + right_time) / 2

        left_point = generate_bezier_cubic(avg_control_points, left_time)
        right_point = generate_bezier_cubic(avg_control_points, right_time)
        center_point = generate_bezier_cubic(avg_control_points, center_time)
        
        draw.point(tuple(center_point), fill='blue')
        
        vec = np.array(right_point - left_point)
        angle_c = np.degrees(np.arctan2(vec[1], vec[0]))
        bbox = font.getbbox(c)

        left_top = center_point - np.array([bbox[2] + bbox[0], bbox[3] + bbox[1]]) / 2
        right_bottom = center_point + np.array([bbox[2] + bbox[0], bbox[3] + bbox[1]]) / 2
        draw.text((left_top[0], left_top[1]), c, font=font, fill=(255, 255, 255, 255))
        # draw.rectangle((left_top[0], left_top[1], right_bottom[0], right_bottom[1]), outline='green')
        # img_c.save(f'draw_glyph_test/glyph_c_{i}.png')
        img_c = img_c.rotate(-angle_c, center=tuple(center_point))
        img = Image.alpha_composite(img, img_c)

    # img_RGB = draw_polygon(width, height, np.concatenate([up_points, down_points[::-1]], axis=0), color='red')
    img = np.expand_dims(np.array(img.convert("1")), axis=2).astype(np.float64)

    return img, angle, (round(center[0] / 2), round(center[1] / 2))
