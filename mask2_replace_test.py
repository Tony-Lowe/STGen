import datetime
import torch
import enum
import os
import cv2
import random
import numpy as np
import re
import json
import argparse
from PIL import ImageFont
from pytorch_lightning import seed_everything
import einops
import time
from PIL import Image


from mllm import GPT4_maskGen, get_pos_n_glyph
from util import (
    check_channels,
    resize_image,
    arr2tensor,
    check_overlap_polygon,
    get_lines,
    count_lines,
    sep_mask,
    draw_pos,
    draw_glyph_curve,
)
from t3_dataset import (
    draw_glyph,
    draw_glyph2,
    draw_glyph3,
    draw_glyph4,
    get_caption_pos,
)


BBOX_MAX_NUM = 8
PLACE_HOLDER = "*"

save_memory = False
load_model = True
max_chars = 20
max_lines = 20
phrase_list = [
    ", content and position of the texts are ",
    ", textual material depicted in the image are ",
    ", texts that says ",
    ", captions shown in the snapshot are ",
    ", with the words of ",
    ", that reads ",
    ", the written materials on the picture: ",
    ", these texts are written on it: ",
    ", captions are ",
    ", content of the text in the graphic is ",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_fp32",
        action="store_true",
        default=False,
        help="Whether or not to use fp32 during inference.",
    )
    parser.add_argument(
        "--no_translator",
        action="store_true",
        default=False,
        help="Whether or not to use the CH->EN translator, which enable input Chinese prompt and cause ~4GB VRAM.",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="font/Arial_Unicode.ttf",
        help="path of a font file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/hugging_face_models/cv_anytext_text_generation_editing/anytext_v1.1.ckpt",
        help="load a specified anytext checkpoint",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./models_yaml/anytext_sd15.yaml",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Results/AdaIN_fuse_replace",
    )
    parser.add_argument(
        "--st",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.00,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.00,
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--l_lambda",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--op_step",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


args = parse_args()
img_save_folder = os.path.join(args.save_path,f"{args.theta}_{args.omega}_step{args.st}->{args.ed}")
infer_params = {
    "model": "damo/cv_anytext_text_generation_editing",
    "model_revision": "v1.1.3",
    "use_fp16": not args.use_fp32,
    "use_translator": not args.no_translator,
    "font_path": args.font_path,
}
if args.model_path:
    infer_params["model_path"] = args.model_path
if load_model:
    # inference = pipeline("my-anytext-task", **infer_params)
    from cldm.model import create_model, load_state_dict
    from cldm_mod.ddim_two_mask_add import myDDIMSampler
    from cldm.ddim_hacked import DDIMSampler
    model = create_model(args.config_yaml).cuda()
    # print(model.cond_stage_model.transformer.text_model.embeddings)
    model.load_state_dict(
        load_state_dict(args.model_path, location="cuda"), strict=True
    )
    # print(model.control_model.input_blocks.state_dict().keys())
    # print(model.control_model.input_blocks[5][1].transformer_blocks[0])
font = ImageFont.truetype("./font/Arial_Unicode.ttf", size=60)


def get_masked_x(hint, H,W):
    ref_img = np.ones((H, W, 3)) * 127.5
    masked_img = ((ref_img.astype(np.float32) / 127.5) - 1.0) * (1 - hint)
    masked_img = np.transpose(masked_img, (2, 0, 1))
    masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
    encoder_posterior = model.encode_first_stage(masked_img[None, ...])
    masked_x = model.get_first_stage_encoding(encoder_posterior).detach()

    return masked_x


def get_latent(prompt, polygons, params, flat_ref=None):
    """
    Args:
        prompt: str, the prompt text
        polygons: list of list of tuple, the list of polygons
        params: dict, the parameters
        flat_ref: dict, with the flat reference tensor in it
    """
    H, W = (params["image_height"], params["image_width"])
    info = {}
    info["glyphs"] = []
    info["polygons"] = polygons
    info["gly_line"] = []
    info["positions"] = []
    info["step_size"] = params["step_size"]
    # info["font"] = font
    texts, prompt = get_lines(prompt)
    info["texts"] = texts
    n_lines = min(len(texts), max_lines)
    info["n_lines"] = [n_lines] * params["image_count"]
    positions = []
    for idx, text in enumerate(texts):
        gly_line = draw_glyph(font, text)
        glyphs, angle, center = draw_glyph3(font, text, polygons[idx], scale=2)
        info["gly_line"] += [arr2tensor(gly_line, params["image_count"])]
        info["glyphs"] += [arr2tensor(glyphs, params["image_count"])]
    for idx, poly in enumerate(polygons):
        positions += [draw_pos(poly, 1.0)]
        info["positions"] += [
            arr2tensor(positions[idx], params["image_count"])
        ]  # shape [batch_size ,1 , 512, 512]
    # get masked_x
    hint = np.sum(positions, axis=0).clip(0, 1)
    masked_x = get_masked_x(hint, H,W)
    # print(hint.shape)
    info["masked_x"] = torch.cat(
        [masked_x for _ in range(params["image_count"])], dim=0
    )
    hint = arr2tensor(hint, params["image_count"])

    if flat_ref is None:
        info["use_masa"] = False
        ddim_sampler = DDIMSampler(model=model)
    else:
        ddim_sampler = myDDIMSampler(
            model,
            ref_lat=flat_ref,
            start_step=params["start_op_step"],
            end_step=params["end_op_step"],
            max_op_step=params["OPTIMIZE_STEPS"],
            loss_alpha=params["alpha"],
            loss_beta=params["beta"],
            add_theta=params["theta"],
            add_omega=params["omega"],
            save_mem=save_memory,
        )
    batch_size = params["image_count"]


    cond = model.get_learned_conditioning(
        dict(
            c_concat=[hint],
            c_crossattn=[[prompt + "," + params["a_prompt"]] * batch_size],
            text_info=info,
        )
    )
    un_cond = model.get_learned_conditioning(
        dict(
            c_concat=[hint],
            c_crossattn=[[params["n_prompt"]] * batch_size],
            text_info=info,
        )
    )
    shape = (4, H // 8, W // 8)
    if save_memory:
        model.low_vram_shift(is_diffusing=True)
    model.control_scales = [params["strength"]] * 13
    samples, intermediates = ddim_sampler.sample(
        params["ddim_steps"],
        batch_size,
        shape,
        cond,
        log_every_t=1,
        verbose=False,
        eta=params["eta"],
        unconditional_guidance_scale=params["cfg_scale"],
        unconditional_conditioning=un_cond,
    )
    # if flat_ref is not None:
    #     for idx, x_inter in enumerate(intermediates["x_inter"]):
    #         # print(idx)
    #         x_inter = model.decode_first_stage(x_inter)
    #         decode_x0 = model.decode_first_stage(intermediates["pred_x0"][idx])
    #         decode_x0 = torch.clamp(decode_x0, -1, 1)
    #         decode_x0 = (decode_x0 + 1.0) / 2.0 * 255  # -1,1 -> 0,255; n, c,h,w
    #         decode_x0 = einops.rearrange(decode_x0, "b c h w -> b h w c").cpu().numpy().clip(0, 255).astype(np.uint8)
    #         x_inter = (einops.rearrange(x_inter, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    #         if not os.path.exists(os.path.join(img_save_folder,"inter")):
    #             os.makedirs(os.path.join(img_save_folder, "inter"))
    #         for i in range(params['image_count']):
    #             cv2.imwrite(os.path.join(img_save_folder, "inter",f"{idx}_{i}.png"), x_inter[i])
    #         if not os.path.exists(os.path.join(img_save_folder, "inter_other")):
    #             os.makedirs(os.path.join(img_save_folder, "inter_other"))
    #         for i in range(params['image_count']):
    #             cv2.imwrite(os.path.join(img_save_folder, "inter_other",f"{idx}_{i}.png"), decode_x0[i])
    info["samples"] = samples
    return info


def inference(input_data, **params):
    prompt = input_data["prompt"]
    seed = input_data["seed"]
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)
    pos_imgs = input_data["draw_pos"]
    flat_pos = input_data["draw_ref"]
    polygons = sep_mask(pos_imgs, sort_radio=params["sort_priority"])
    flat_polygons = sep_mask(flat_pos, sort_radio=params["sort_priority"])
    tic = time.time()
    flat_ref = get_latent(prompt, polygons=flat_polygons, params=params)
    info = get_latent(prompt, polygons, params, flat_ref=flat_ref)
    samples = info["samples"]

    cost = (time.time() - tic) * 1000.0
    if save_memory:
        model.low_vram_shift(is_diffusing=False)
    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    results = [x_samples[i] for i in range(params["image_count"])]
    results += [cost]
    return results,info


def save_images(img_list, folder, prompt=""):
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H_%M_%S")
    for idx, img in enumerate(img_list):
        image_number = idx + 1
        filename = f"{prompt}_{date_str}_{time_str}_{image_number}.png"
        save_path = os.path.join(folder, filename)
        cv2.imwrite(save_path, img[..., ::-1])


def process(
    prompt,
    draw_img,
    ref_mask,
    sort_radio="↕",
    ori_img=None,
    img_count=1,
    ddim_steps=20,
    w=512,
    h=512,
    strength=1,
    cfg_scale=9,
    seed=-1,
    eta=0,
    a_prompt="best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks",
    n_prompt="low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
    times=1,
    times1=1,
    op_step=1,
    start_op_step=0,
    end_op_step=0,
    loss_alpha=0,
    loss_beta=0,
    loss_lambda=0.0,
    add_theta=0.35,
    add_omega=0.3,
    use_gpt=False,
    step_size=3.75,
    font_path="font/Arial_Unicode.ttf",
):
    torch.cuda.empty_cache()
    # Text Generation
    # create pos_imgs
    if draw_img is not None:
        pos_imgs = 255 - draw_img
    else:
        pos_imgs = np.zeros((w, h, 1))
    if ref_mask is not None:
        ref_pos = 255 - ref_mask
    else:
        ref_pos = np.zeros((w, h, 1))
    # center = (pos_imgs.shape[0] // 2, pos_imgs.shape[1] // 2)
    # rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rot_pos_imgs = cv2.warpAffine(pos_imgs,rotMat,pos_imgs.shape[1:-1])
    # print(times)
    params = {
        "sort_priority": sort_radio,
        "image_count": img_count,
        "ddim_steps": ddim_steps,
        "image_width": w,
        "image_height": h,
        "strength": strength,
        "cfg_scale": cfg_scale,
        "eta": eta,
        "a_prompt": a_prompt,
        "n_prompt": n_prompt,
        "times": times,
        "times1": times1,
        "start_op_step": int(start_op_step),
        "end_op_step": int(end_op_step),
        "OPTIMIZE_STEPS": int(op_step),
        "alpha": loss_alpha,
        "beta": loss_beta,
        "lambda": loss_lambda,
        "theta": add_theta,
        "omega": add_omega,
        "use_gpt": use_gpt,
        "step_size": step_size,
        "Font_path": font_path,
    }
    input_data = {
        "prompt": prompt,
        "seed": seed,
        "draw_pos": pos_imgs,  # numpy (w, h, 1)
        "draw_ref": ref_pos,
        "ori_image": ori_img,
    }

    results,info = inference(input_data, **params)
    time = results.pop()
    return results,info


if __name__ == "__main__":
    import logging

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H_%M_%S")
    if not os.path.exists(img_save_folder):
        os.makedirs(img_save_folder)
    logging.basicConfig(
        filename=f"{img_save_folder}/{date_str}_{time_str}_args.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    logging.info(args)
    font_lists = [
        "font/Arial_Unicode.ttf",
    ]
    seeds = {
        'A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"': 35621187,
        'A meticulously designed logo, a minimalist brain, stick drawing style, simplistic style,  refined with minimal strokes, black and white color, white background,  futuristic sense, exceptional design, logo name is "NextAI"': 2563689,
        'A raccoon stands in front of the blackboard with the words "Deep Learning" written on it': 33789703,
        'A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it': 83866922,
        
        # 'a photo of a lovely couple with a banner writes "新婚快乐"':42,
    }
    mask_lists = [
        # """./masks/a photo of a lovely couple with a banner writes "新婚快乐"_1_1.png""",
        # """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1_1_1_1.png""",
        # """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1_1_1.png""",
        # """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1_1.png""",
        # """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1.png""",
        # """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman".png""",
        """Result/ori/masks/A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it_1_1.png""",
        """Result/ori/masks/A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it_1.png""",
        # """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it_1_1_1.png""",
        # """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it_1_1.png""",
        # """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it_1.png""",
        # """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it.png""",
    ]
    ref_mask_list = {
        'A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"': "example_images/gen18.png",
        'A raccoon stands in front of the blackboard with the words "Deep Learning" written on it': "example_images/gen17.png",
        'A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it': "example_images/gen7.png",
    }
    seed_lists = []
    for mask in mask_lists:
        prompt = mask.split("/")[-1].split(".")[0].replace("_1", "")
        draw_img = cv2.imread(mask)
        ref_mask = cv2.imread(ref_mask_list[prompt])
        for font_path in font_lists:
            results,_ = process(
                prompt,
                draw_img,
                ref_mask=ref_mask,
                seed=seeds[prompt],
                font_path=font_path,
                img_count=4,
                start_op_step=args.st,
                end_op_step=args.ed,
                loss_alpha=args.alpha,
                loss_beta=args.beta,
                loss_lambda=args.l_lambda,
                step_size=args.step_size,
                add_theta=args.theta,
                add_omega=args.omega,
                op_step=args.op_step,
            )
            save_images(results, img_save_folder, prompt.replace(" ", "_"))
            print(f"Done, result images are saved in: {img_save_folder}")
