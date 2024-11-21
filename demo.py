import datetime
import os
from matplotlib.pyplot import draw
import torch
from modelscope.pipelines import pipeline
import cv2
import gradio as gr
import random
import numpy as np
import re
import json
import argparse
from PIL import ImageFont
from gradio.components import Component
from pytorch_lightning import seed_everything
import einops
import time
from PIL import Image

from util import (
    check_channels,
    resize_image,
    save_images,
    arr2tensor,
    draw_glyph_curve,
    sep_mask,
    draw_pos,
    count_lines,
    get_lines,
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
        "--save_mem",
        action="store_true",
        default=False,
        help="Whether or not to memory shift",
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
        # default="your path to the anytext checkpoint",
        default="/data/hugging_face_models/cv_anytext_text_generation_editing/anytext_v1.1.ckpt",
        help="load a specified anytext checkpoint",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./models_yaml/anytext_sd15.yaml",  # wo _masa
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Result/for_paper",
        help="load a specified anytext checkpoint",
    )
    args = parser.parse_args()
    return args


args = parse_args()
save_memory = args.save_mem
img_save_folder = args.save_path
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
    from cldm.ddim_hacked import DDIMSampler
    from cldm_mod.ddim_two_mask_add import myDDIMSampler
    import yaml

    with open(args.config_yaml, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if (
        "start_step" in config["model"]["params"]["control_stage_config"]["params"]
        and "start_step" in config["model"]["params"]["control_stage_config"]["params"]
    ):
        masa_in_model = True
    else:
        masa_in_model = False

    model = create_model(args.config_yaml).cuda()
    # print(model.cond_stage_model.transformer.text_model.embeddings)
    model.load_state_dict(
        load_state_dict(args.model_path, location="cuda"), strict=True
    )
    # print(model.control_model.input_blocks.state_dict().keys())
    # print(model.control_model.input_blocks[5][1].transformer_blocks[0])

font = ImageFont.truetype("./font/Arial_Unicode.ttf", size=60)


def generate_rectangles(w, h, n, max_trys=200):
    img = np.zeros((h, w, 1), dtype=np.uint8)
    rectangles = []
    attempts = 0
    n_pass = 0
    low_edge = int(max(w, h) * 0.3 if n <= 3 else max(w, h) * 0.2)  # ~150, ~100
    while attempts < max_trys:
        rect_w = min(np.random.randint(max((w * 0.5) // n, low_edge), w), int(w * 0.8))
        ratio = np.random.uniform(4, 10)
        rect_h = max(low_edge, int(rect_w / ratio))
        rect_h = min(rect_h, int(h * 0.8))
        # gen rotate angle
        rotation_angle = 0
        rand_value = np.random.rand()
        if rand_value < 0.7:
            pass
        elif rand_value < 0.8:
            rotation_angle = np.random.randint(0, 40)
        elif rand_value < 0.9:
            rotation_angle = np.random.randint(140, 180)
        else:
            rotation_angle = np.random.randint(85, 95)
        # rand position
        x = np.random.randint(0, w - rect_w)
        y = np.random.randint(0, h - rect_h)
        # get vertex
        rect_pts = cv2.boxPoints(
            ((rect_w / 2, rect_h / 2), (rect_w, rect_h), rotation_angle)
        )
        rect_pts = np.int32(rect_pts)
        # move
        rect_pts += (x, y)
        # check boarder
        if (
            np.any(rect_pts < 0)
            or np.any(rect_pts[:, 0] >= w)
            or np.any(rect_pts[:, 1] >= h)
        ):
            attempts += 1
            continue
        # check overlap
        if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles):
            attempts += 1
            continue
        n_pass += 1
        cv2.fillPoly(img, [rect_pts], 255)
        rectangles.append(rect_pts)
        if n_pass == n:
            break
    print("attempts:", attempts)
    if len(rectangles) != n:
        raise gr.Error(
            f"Failed in auto generate positions after {attempts} attempts, try again!"
        )
    return img


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


def draw_rects(width, height, rects):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for rect in rects:
        x1 = int(rect[0] * width)
        y1 = int(rect[1] * height)
        w = int(rect[2] * width)
        h = int(rect[3] * height)
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, -1)
    return img


def get_masked_x(hint, H, W):
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
        if params["curve"]:
            glyphs, angle, center = draw_glyph_curve(font, text, polygons[idx], scale=2)
        else:
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
    masked_x = get_masked_x(hint, H, W)
    # print(hint.shape)
    info["masked_x"] = torch.cat(
        [masked_x for _ in range(params["image_count"])], dim=0
    )
    hint = arr2tensor(hint, params["image_count"])

    if flat_ref is None:
        info["use_masa"] = False
        ddim_sampler = DDIMSampler(model=model)
        batch_size = params["image_count"]
    else:
        ddim_sampler = myDDIMSampler(
            model,
            ref_lat=flat_ref,
            start_step=params["start_op_step"],
            end_step=params["end_op_step"],
            max_op_step=params["OPTIMIZE_STEPS"],
            loss_alpha=0.0,
            loss_beta=0.0,
            # loss_alpha=params["alpha"],
            # loss_beta=params["beta"],
            add_theta=params["theta"],
            add_omega=params["omega"],
            save_mem=save_memory,
            use_masa=masa_in_model,
        )
        if masa_in_model:
            batch_size = params["image_count"] * 2
            info["flat_glyphs"] = flat_ref["glyphs"]
            info["flat_positions"] = flat_ref["positions"]
            info["flat_masked_x"] = flat_ref["masked_x"]
            info["use_masa"] = True
            info["attn_mask"] = [
                torch.cat([i.unsqueeze(0) for i in info["flat_positions"]], dim=0)
                .sum(dim=0, keepdim=True)
                .squeeze(0),
                torch.cat([i.unsqueeze(0) for i in info["positions"]], dim=0)
                .sum(dim=0, keepdim=True)
                .squeeze(0),
            ]
        else:
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
    now = datetime.datetime.now()
    date_str = now.strftime("%m-%d-%H-%M")
    # if flat_ref is not None:
    #     for idx, x_inter in enumerate(intermediates["x_inter"]):
    #         # print(idx)
    #         # x_inter = model.decode_first_stage(x_inter)
    #         decode_x0 = model.decode_first_stage(intermediates["pred_x0"][idx])
    #         decode_x0 = torch.clamp(decode_x0, -1, 1)
    #         decode_x0 = (decode_x0 + 1.0) / 2.0 * 255  # -1,1 -> 0,255; n, c,h,w
    #         decode_x0 = einops.rearrange(decode_x0, "b c h w -> b h w c").cpu().numpy().clip(0, 255).astype(np.uint8)
    #         x_inter = (einops.rearrange(x_inter, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    #         # if not os.path.exists(os.path.join(img_save_folder,f"{prompt.replace(' ','_')}","inter")):
    #         #     os.makedirs(os.path.join(img_save_folder, "inter"))
    #         # for i in range(params['image_count']):
    #         #     cv2.imwrite(os.path.join(img_save_folder, "inter",f"{idx}_{i}.png"), x_inter[i])
    #         if not os.path.exists(os.path.join(img_save_folder, "pred_x0")):
    #             os.makedirs(os.path.join(img_save_folder, "pred_x0"))
    #         for i in range(params['image_count']):
    #             cv2.imwrite(os.path.join(img_save_folder, "pred_x0",f"{idx}_{i}.png"), decode_x0[i])
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
    return results


def process(
    mode,
    prompt,
    pos_radio,
    sort_radio,
    revise_pos,
    base_model_path,
    lora_path_ratio,
    show_debug,
    draw_img,
    rect_img,
    img_count,
    ddim_steps,
    w,
    h,
    strength,
    cfg_scale,
    seed,
    eta,
    a_prompt,
    n_prompt,
    times,
    times1,
    step_size,
    lamb,
    op_steps,
    start_op_step,  # -1
    end_op_step,  # -1
    add_theta,
    add_omega,
    draw_ref,
    use_curve,
    *rect_list,
):
    torch.cuda.empty_cache()
    n_lines = count_lines(prompt)
    # Text Generation
    if mode == "gen":
        # create pos_imgs
        if pos_radio == "Manual-draw(手绘)":
            if draw_img is not None:
                pos_imgs = 255 - draw_img["image"]
                if "mask" in draw_img:
                    pos_imgs = pos_imgs.astype(np.float32) + draw_img["mask"][
                        ..., 0:3
                    ].astype(np.float32)
                    pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
            else:
                pos_imgs = np.zeros((w, h, 1))
            if draw_ref is not None:
                ref_pos = 255 - draw_ref["image"]
                if "mask" in draw_ref:
                    ref_pos = ref_pos.astype(np.float32) + draw_ref["mask"][
                        ..., 0:3
                    ].astype(np.float32)
                    ref_pos = ref_pos.clip(0, 255).astype(np.uint8)
                else:
                    ref_pos = np.zeros((w, h, 1))

        elif pos_radio == "Manual-rect(拖框)":
            rect_check = rect_list[:BBOX_MAX_NUM]
            rect_xywh = rect_list[BBOX_MAX_NUM:]
            checked_rects = []
            for idx, c in enumerate(rect_check):
                if c:
                    _xywh = rect_xywh[4 * idx : 4 * (idx + 1)]
                    checked_rects += [_xywh]
            pos_imgs = draw_rects(w, h, checked_rects)
        elif pos_radio == "Auto-rand(随机)":
            pos_imgs = generate_rectangles(w, h, n_lines, max_trys=500)
    # Text Editing
    elif mode == "edit":
        # revise_pos = False  # disable pos revise in edit mode
        # if ref_img is None or ori_img is None:
        #     raise gr.Error("No reference image, please upload one for edit!")
        # edit_image = ori_img.clip(1, 255)  # for mask reason
        # edit_image = check_channels(edit_image)
        # edit_image = resize_image(edit_image, max_length=768)
        # h, w = edit_image.shape[:2]
        # if (
        #     isinstance(ref_img, dict)
        #     and "mask" in ref_img
        #     and ref_img["mask"].mean() > 0
        # ):
        #     pos_imgs = 255 - edit_image
        #     edit_mask = cv2.resize(ref_img["mask"][..., 0:3], (w, h))
        #     pos_imgs = pos_imgs.astype(np.float32) + edit_mask.astype(np.float32)
        #     pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
        # else:
        #     if isinstance(ref_img, dict) and "image" in ref_img:
        #         ref_img = ref_img["image"]
        #     pos_imgs = 255 - ref_img  # example input ref_img is used as pos
        pass
    if not os.path.exists(os.path.join(img_save_folder, "masks")):
        os.makedirs(os.path.join(img_save_folder, "masks"))
    mask_save = os.path.join(img_save_folder, "masks", f"{prompt}.png")
    while os.path.exists(mask_save):
        mask_save = mask_save.replace(".png", "_1.png")
    cv2.imwrite(mask_save, 255 - pos_imgs[..., ::-1])
    # center = (pos_imgs.shape[0] // 2, pos_imgs.shape[1] // 2)
    # rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rot_pos_imgs = cv2.warpAffine(pos_imgs,rotMat,pos_imgs.shape[1:-1])
    # print(times)
    params = {
        "mode": mode,
        "sort_priority": sort_radio,
        "show_debug": show_debug,
        "revise_pos": revise_pos,
        "image_count": img_count,
        "ddim_steps": ddim_steps,
        "image_width": w,
        "image_height": h,
        "strength": strength,
        "cfg_scale": cfg_scale,
        "eta": eta,
        "a_prompt": a_prompt,
        "n_prompt": n_prompt,
        "base_model_path": base_model_path,
        "lora_path_ratio": lora_path_ratio,
        "times": times,
        "times1": times1,
        "step_size": step_size,
        "lambda": lamb,
        "start_op_step": int(start_op_step),
        "end_op_step": int(end_op_step),
        "OPTIMIZE_STEPS": int(op_steps),
        "theta": add_theta,
        "omega": add_omega,
        "curve": use_curve,
    }
    input_data = {
        "prompt": prompt,
        "seed": seed,
        "draw_pos": pos_imgs,  # numpy (w, h, 1)
        "draw_ref": ref_pos,
        # "ori_image": ori_img,
    }

    results = inference(input_data, **params)
    time = results.pop()
    # if rtn_code >= 0:
    save_images(results, img_save_folder)
    print(f"Done, result images are saved in: {img_save_folder}")
    #     if rtn_warning:
    #         gr.Warning(rtn_warning)
    # else:
    #     raise gr.Error(rtn_warning)
    return results, gr.Markdown(f"Time: {time}", visible=show_debug)


def create_canvas(w=512, h=512, c=3, line=5):
    image = np.full((h, w, c), 200, dtype=np.uint8)
    for i in range(h):
        if i % (w // line) == 0:
            image[i, :, :] = 150
    for j in range(w):
        if j % (w // line) == 0:
            image[:, j, :] = 150
    image[h // 2 - 8 : h // 2 + 8, w // 2 - 8 : w // 2 + 8, :] = [200, 0, 0]
    return image


def resize_w(w, img1, img2):
    if isinstance(img2, dict):
        img2 = img2["image"]
    return [cv2.resize(img1, (w, img1.shape[0])), cv2.resize(img2, (w, img2.shape[0]))]


def resize_h(h, img1, img2):
    if isinstance(img2, dict):
        img2 = img2["image"]
    return [cv2.resize(img1, (img1.shape[1], h)), cv2.resize(img2, (img2.shape[1], h))]


def mask_upload(mask_path):
    mask_pic = cv2.imread(mask_path.name)
    return mask_pic


is_t2i = "true"
block = gr.Blocks(css="style.css", theme=gr.themes.Soft()).queue()

with open("javascript/bboxHint.js", "r") as file:
    value = file.read()
escaped_value = json.dumps(value)

with block:
    block.load(
        fn=None,
        _js=f"""() => {{
               const script = document.createElement("script");
               const text =  document.createTextNode({escaped_value});
               script.appendChild(text);
               document.head.appendChild(script);
               }}""",
    )
    with gr.Row(variant="compact"):
        with gr.Column() as left_part:
            pass
        with gr.Column():
            result_gallery = gr.Gallery(
                label="Result(结果)",
                show_label=True,
                preview=True,
                columns=2,
                allow_preview=True,
                height=600,
            )
            result_info = gr.Markdown("", visible=False)
            with gr.Row():
                gr.Markdown("")
                use_curve = gr.Checkbox(
                    label="Use Bezier glyph draw(是否使用贝塞尔曲线拟合mask)",
                    value=False,
                )
                gr.Markdown("")
            with gr.Row():
                gr.Markdown("")
                run_gen = gr.Button(value="Run(运行)!", scale=0.3, elem_classes="run")
                gr.Markdown("")
            with gr.Row():
                gr.Markdown("")
                run_edit = gr.Button(
                    value="Run Edit(运行编辑)!", scale=0.3, elem_classes="run",visible=False
                )
                gr.Markdown("")
        with left_part:
            with gr.Accordion("🛠Parameters(参数)", open=False):
                with gr.Row(variant="compact"):
                    img_count = gr.Slider(
                        label="Image Count(图片数)",
                        minimum=1,
                        maximum=12,
                        value=1,
                        step=1,
                    )
                    ddim_steps = gr.Slider(
                        label="Steps(步数)", minimum=1, maximum=100, value=20, step=1
                    )
                with gr.Row(variant="compact"):
                    image_width = gr.Slider(
                        label="Image Width(宽度)",
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                    )
                    image_height = gr.Slider(
                        label="Image Height(高度)",
                        minimum=256,
                        maximum=768,
                        value=512,
                        step=64,
                    )
                with gr.Row(variant="compact"):
                    strength = gr.Slider(
                        label="Strength(控制力度)",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.01,
                    )
                    cfg_scale = gr.Slider(
                        label="CFG-Scale(CFG强度)",
                        minimum=0.1,
                        maximum=30.0,
                        value=9.0,
                        step=0.1,
                    )
                with gr.Row(variant="compact"):
                    seed = gr.Slider(
                        label="Seed(种子数)",
                        minimum=-1,
                        maximum=99999999,
                        step=1,
                        randomize=False,
                        value=-1,
                    )
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                with gr.Row(variant="compact"):
                    show_debug = gr.Checkbox(label="Show Debug(调试信息)", value=False)
                    gr.Markdown(
                        '<span style="color:silver;font-size:12px">whether show glyph image and debug information in the result(是否在结果中显示glyph图以及调试信息)</span>'
                    )
                with gr.Row(variant="compact"):
                    Clip_times = gr.Number(
                        label="Clip Times(Clip Embedding增强倍数)", value=1.0
                    )
                    Clip_times1 = gr.Number(
                        label="Clip Times(内容外的Clip Embedding增强倍数)", value=1.0
                    )
                a_prompt = gr.Textbox(
                    label="Added Prompt(附加提示词)",
                    value="best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks",
                )
                n_prompt = gr.Textbox(
                    label="Negative Prompt(负向提示词)",
                    value="low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
                )
                base_model_path = gr.Textbox(label="Base Model Path(基模地址)")
                lora_path_ratio = gr.Textbox(
                    label="LoRA Path and Ratio(lora地址和比例)"
                )

            with gr.Accordion("🛠Optimization Parameters(优化参数)", open=False):
                with gr.Row(variant="compact"):
                    step_size = gr.Number(label="Step Size(优化步长)", value=0,visible=False)
                    lamd = gr.Number(label="Lambda(loss系数)", value=0,visible=False)
                    op_step = gr.Number(label="Optimize Steps(优化步数)", value=1)
                with gr.Row(variant="compact"):
                    start_op_step = gr.Number(
                        label="Start Optimize Step(结束优化的inference步数)", value=0
                    )
                    end_op_step = gr.Number(
                        label="End Optimize Step(结束优化的inference步数)", value=0
                    )
                with gr.Row(variant="compact"):
                    add_theta = gr.Number(label="Lambda", value=0.5)
                    add_omega = gr.Number(label="Rho", value=0.45)

            prompt = gr.Textbox(label="Prompt(提示词)")
            with gr.Tabs() as tab_modes:
                with gr.Tab(
                    "🖼Text Generation(文字生成)", elem_id="MD-tab-t2i"
                ) as mode_gen:
                    pos_radio = gr.Radio(
                        ["Manual-draw(手绘)", "Manual-rect(拖框)", "Auto-rand(随机)"],
                        value="Manual-draw(手绘)",
                        label="Pos-Method(位置方式)",
                        info="choose a method to specify text positions(选择方法用于指定文字位置).",
                    )
                    with gr.Row():
                        sort_radio = gr.Radio(
                            ["↕", "↔"],
                            value="↕",
                            label="Sort Position(位置排序)",
                            info="position sorting priority(位置排序时的优先级)",
                        )
                        revise_pos = gr.Checkbox(
                            label="Revise Position(修正位置)", value=False
                        )
                    upload_button = gr.UploadButton(
                        "Click to upload Mask", file_types=["image"]
                    )
                    upload_ref_button = gr.UploadButton(
                        "Click to upload ref Mask", file_types=["image"]
                    )
                    with gr.Row(variant="compact"):
                        rect_cb_list: list[Component] = []
                        rect_xywh_list: list[Component] = []
                        for i in range(BBOX_MAX_NUM):
                            e = gr.Checkbox(
                                label=f"{i}", value=False, visible=False, min_width="10"
                            )
                            x = gr.Slider(
                                label="x",
                                value=0.4,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.0001,
                                elem_id=f"MD-t2i-{i}-x",
                                visible=False,
                            )
                            y = gr.Slider(
                                label="y",
                                value=0.4,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.0001,
                                elem_id=f"MD-t2i-{i}-y",
                                visible=False,
                            )
                            w = gr.Slider(
                                label="w",
                                value=0.2,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.0001,
                                elem_id=f"MD-t2i-{i}-w",
                                visible=False,
                            )
                            h = gr.Slider(
                                label="h",
                                value=0.2,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.0001,
                                elem_id=f"MD-t2i-{i}-h",
                                visible=False,
                            )
                            x.change(
                                fn=None,
                                inputs=x,
                                outputs=x,
                                _js=f'v => onBoxChange({is_t2i}, {i}, "x", v)',
                                show_progress=False,
                                queue=False,
                            )
                            y.change(
                                fn=None,
                                inputs=y,
                                outputs=y,
                                _js=f'v => onBoxChange({is_t2i}, {i}, "y", v)',
                                show_progress=False,
                                queue=False,
                            )
                            w.change(
                                fn=None,
                                inputs=w,
                                outputs=w,
                                _js=f'v => onBoxChange({is_t2i}, {i}, "w", v)',
                                show_progress=False,
                                queue=False,
                            )
                            h.change(
                                fn=None,
                                inputs=h,
                                outputs=h,
                                _js=f'v => onBoxChange({is_t2i}, {i}, "h", v)',
                                show_progress=False,
                                queue=False,
                            )

                            e.change(
                                fn=None,
                                inputs=e,
                                outputs=e,
                                _js=f"e => onBoxEnableClick({is_t2i}, {i}, e)",
                                queue=False,
                            )
                            rect_cb_list.extend([e])
                            rect_xywh_list.extend([x, y, w, h])

                    rect_img = gr.Image(
                        value=create_canvas(),
                        label="Rext Position(方框位置)",
                        elem_id="MD-bbox-rect-t2i",
                        show_label=False,
                        visible=False,
                    )
                    rect_ref = gr.Image(
                        value=create_canvas(),
                        label="Reference Rext Position(方框位置)",
                        elem_id="MD-bbox-rect-t2i",
                        show_label=False,
                        visible=False,
                    )
                    draw_img = gr.Image(
                        value=create_canvas(),
                        label="Draw Position(绘制位置)",
                        visible=True,
                        tool="sketch",
                        show_label=False,
                        brush_radius=100,
                    )
                    draw_ref = gr.Image(
                        value=create_canvas(),
                        label=" Reference Draw Position(参考mask绘制位置)",
                        visible=True,
                        tool="sketch",
                        show_label=False,
                        brush_radius=100,
                    )

                    def re_draw():
                        return [
                            gr.Image(value=create_canvas(), tool="sketch"),
                            gr.Slider(value=512),
                            gr.Slider(value=512),
                        ]

                    draw_img.clear(re_draw, None, [draw_img, image_width, image_height])
                    image_width.release(
                        resize_w,
                        [image_width, rect_img, draw_img],
                        [rect_img, draw_img],
                    )
                    image_height.release(
                        resize_h,
                        [image_height, rect_img, draw_img],
                        [rect_img, draw_img],
                    )

                    draw_ref.clear(re_draw, None, [draw_ref, image_width, image_height])
                    image_width.release(
                        resize_w,
                        [image_width, rect_ref, draw_ref],
                        [rect_ref, draw_ref],
                    )
                    image_height.release(
                        resize_h, [image_height, draw_ref], [rect_ref, draw_ref]
                    )

                    def change_options(selected_option):
                        return [
                            gr.Checkbox(visible=selected_option == "Manual-rect(拖框)")
                        ] * BBOX_MAX_NUM + [
                            gr.Image(visible=selected_option == "Manual-rect(拖框)"),
                            gr.Image(visible=selected_option == "Manual-draw(手绘)"),
                            gr.Radio(visible=selected_option != "Auto-rand(随机)"),
                            gr.Checkbox(value=selected_option == "Auto-rand(随机)"),
                        ]

                    pos_radio.change(
                        change_options,
                        pos_radio,
                        rect_cb_list + [rect_img, draw_img, sort_radio, revise_pos],
                        show_progress=False,
                        queue=False,
                    )
                    # with gr.Row():
                    #     gr.Markdown("")
                    #     run_gen = gr.Button(
                    #         value="Run(运行)!", scale=0.3, elem_classes="run"
                    #     )
                    #     gr.Markdown("")

                    def exp_gen_click():
                        return [
                            gr.Slider(value=512),
                            gr.Slider(value=512),
                        ]  # all examples are 512x512, refresh draw_img

                    with gr.Tab("English Examples"):
                        exp_gen_en = gr.Examples(
                            [
                                [
                                    'A playful cartoon-style illustration of a cat wearing sunglasses with a speech bubble saying "Too Cool"',
                                    "example_images/33.png",
                                    "Manual-draw(手绘)",
                                    "↕",
                                    False,
                                    1,
                                    20001031,
                                    "example_images/33_ref.png",
                                    0.5,
                                    0.45,
                                    0,
                                    0,
                                    False,
                                ],
                                [
                                    'A vibrant street art mural in a lively city square, showcasing "CVPR" in bold, artistic graffiti surrounded by other cheerful designs and bright splashes of color.',
                                    "example_images/10.png",
                                    "Manual-draw(手绘)",
                                    "↕",
                                    False,
                                    4,
                                    37873140,
                                    "example_images/10_ref.jpg",
                                    0.5,
                                    2,
                                    0,
                                    0,
                                    False,
                                ],
                                [
                                    'A forest shrouded in mysterious morning mist, with a playful unicorn frolicking among glowing flowers, trees sparkling with soft light, and the words "Mystic Realm" floating joyfully in the air',
                                    "example_images/1.jpg",
                                    "Manual-draw(手绘)",
                                    "↕",
                                    False,
                                    4,
                                    61317567,
                                    "example_images/1_ref.jpg",
                                    0.5,
                                    0.5,
                                    0,
                                    10,
                                    True,
                                ],
                                [
                                    "An exquisite watch sits elegantly, with its detailed design and clear hands visible on the dial. At the bottom, the text “海鸥表” is presented in a downward arc, seamlessly blending into the image and echoing the watch's elegant style.",
                                    "example_images/24.png",
                                    "Manual-draw(手绘)",
                                    "↕",
                                    False,
                                    4,
                                    63992536,
                                    "example_images/24_ref.png",
                                    0.3,
                                    0.8,
                                    0,
                                    10,
                                    True,
                                ],
                                [
                                    'A steaming cup of coffee on a cozy wooden table, with delicate latte art on top spelling out the words "Good" and "Morning" in creamy, frothy letters.',
                                    "example_images/25.png",
                                    "Manual-draw(手绘)",
                                    "↕",
                                    False,
                                    4,
                                    18843293,
                                    "example_images/25_ref.png",
                                    0.3,
                                    0.8,
                                    0,
                                    10,
                                    True,
                                ],
                                [
                                    'A romantic dinner table set for two, with candles and flowers arranged in a V shape, displaying "Forever" and "Love".',
                                    "example_images/13.png",
                                    "Manual-draw(手绘)",
                                    "↔",
                                    False,
                                    4,
                                    17723880,
                                    "example_images/13_ref.jpg",
                                    0.5,
                                    2,
                                    0,
                                    10,
                                    False,
                                ],
                                [
                                    'On a late night under a sky full of stars, a child walks along the road,with the floating lyrics "Twinkle" "Twinkle" "Little Star" arranged in an S-shape.',
                                    "example_images/17.png",
                                    "Manual-draw(手绘)",
                                    "↕",
                                    False,
                                    4,
                                    -1,
                                    "example_images/17_ref.png",
                                    0.55,
                                    2.0,
                                    0,
                                    10,
                                    False,
                                ],
                            ],
                            [
                                prompt,
                                draw_img,
                                pos_radio,
                                sort_radio,
                                revise_pos,
                                img_count,
                                seed,
                                draw_ref,
                                add_theta,
                                add_omega,
                                start_op_step,
                                end_op_step,
                                use_curve,
                            ],
                            examples_per_page=5,
                            label="",
                        )
                        exp_gen_en.dataset.click(
                            exp_gen_click, None, [image_width, image_height]
                        )
    ips = [
        prompt,
        pos_radio,
        sort_radio,
        revise_pos,
        base_model_path,
        lora_path_ratio,
        show_debug,
        draw_img,
        rect_img,
        img_count,
        ddim_steps,
        image_width,
        image_height,
        strength,
        cfg_scale,
        seed,
        eta,
        a_prompt,
        n_prompt,
        Clip_times,
        Clip_times1,
        step_size,
        lamd,
        op_step,
        start_op_step,
        end_op_step,
        add_theta,
        add_omega,
        draw_ref,
        use_curve,
        *(rect_cb_list + rect_xywh_list),
    ]
    run_gen.click(
        fn=process,
        inputs=[gr.State("gen")] + ips,
        outputs=[result_gallery, result_info],
    )
    run_edit.click(
        fn=process,
        inputs=[gr.State("edit")] + ips,
        outputs=[result_gallery, result_info],
    )
    upload_button.upload(mask_upload, inputs=[upload_button], outputs=[draw_img])
    upload_ref_button.upload(
        mask_upload, inputs=[upload_ref_button], outputs=[draw_ref]
    )


block.launch(
    server_name="0.0.0.0",  # if os.getenv("GRADIO_LISTEN", "") != "" else "127.0.0.1",
    share=False,
    root_path=(
        f"/{os.getenv('GRADIO_PROXY_PATH')}" if os.getenv("GRADIO_PROXY_PATH") else ""
    ),
)
