import nevergrad as ng
import clip
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import torch
import cv2
import numpy as np
from easydict import EasyDict as edict
import logging
import math
from PIL import Image
from einops import rearrange

from mask2_replace_test import process
from cldm.recognizer import TextRecognizer, crop_image
from eval.eval_dgocr import get_ld, pre_process

seeds = {
    'A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"': 35621187,
    'A meticulously designed logo, a minimalist brain, stick drawing style, simplistic style,  refined with minimal strokes, black and white color, white background,  futuristic sense, exceptional design, logo name is "NextAI"': 2563689,
    'A raccoon stands in front of the blackboard with the words "Deep Learning" written on it': 33789703,
    'A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it': 83866922,
}
mask_lists = [
    """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1_1_1_1.png""",
    """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1_1_1.png""",
    """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1_1.png""",
    """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"_1.png""",
    """Result/ori/masks/A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman".png""",
    """Result/ori/masks/A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it_1_1.png""",
    """Result/ori/masks/A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it_1.png""",
    """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it_1_1_1.png""",
    """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it_1_1.png""",
    """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it_1.png""",
    """Result/ori/masks/A raccoon stands in front of the blackboard with the words "Deep Learning" written on it.png""",
]
ref_mask_list = {
    'A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall, titled "Snowman"': "example_images/gen18.png",
    'A raccoon stands in front of the blackboard with the words "Deep Learning" written on it': "example_images/gen17.png",
    'A nice drawing in pencil of Michael Jackson,  with the words "Micheal" and "Jackson" written on it': "example_images/gen7.png",
}
st = 0
ed = 0
op_step = 1
step_size = 0
rec_char_dict_path = os.path.join("./ocr_weights", "en_dict.txt")
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("mask2_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Start NeverGrad Optimization!")


def ocr_acc(imgs, info):
    bsz = len(imgs)
    predictor = pipeline(
        Tasks.ocr_recognition, model="damo/cv_convnextTiny_ocr-recognition-general_damo"
    )
    rec_image_shape = "3, 48, 320"
    args = edict()
    args.rec_image_shape = rec_image_shape
    args.rec_char_dict_path = rec_char_dict_path
    args.rec_batch_num = 1
    args.use_fp16 = False
    text_recognizer = TextRecognizer(args, None)

    texts = info["texts"]
    positions = info["positions"]
    sen_acc = []
    edit_dist = []

    for b in range(bsz):
        n_lines = info["n_lines"][b]
        gt_texts = texts
        pred_texts = []
        img = torch.from_numpy(imgs[b]).permute(2, 0, 1).float()  # C,H,W
        for k in range(n_lines):
            pos = info["positions"][k][b] * 255
            pos = rearrange(pos, "c h w -> h w c")
            np_pos = pos.detach().cpu().numpy().astype(np.uint8)
            pred_text = crop_image(img, np_pos)
            pred_texts += [pred_text]
        if n_lines > 0:
            pred_texts = pre_process(pred_texts, rec_image_shape)
            preds_all = []
            for idx, pt in enumerate(pred_texts):
                rst = predictor(pt)
                preds_all += [rst["text"][0]]
            for k in range(len(preds_all)):
                pred_text = preds_all[k]
                gt_order = [
                    text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1)
                    for m in gt_texts[k]
                ]
                pred_order = [
                    text_recognizer.char2id.get(m, len(text_recognizer.chars) - 1)
                    for m in pred_text
                ]
                if pred_text == gt_texts[k]:
                    sen_acc += [1]
                else:
                    sen_acc += [0]
                edit_dist += [get_ld(pred_order, gt_order)]
    return np.array(sen_acc).mean()


def clip_score(prompt, imgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bsz = len(imgs)
    total = 0
    model, preprocess = clip.load(
        "/data/checkpoints/ViT-B-32.pt", device=device, jit=False
    )
    text = clip.tokenize(prompt).to(device)
    for b in range(bsz):
        img = Image.fromarray(imgs[b].astype("uint8")).convert("RGB")
        # print(img.size)
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Calculate the cosine similarity between the image and text features
            cosine_similarity = (image_features @ text_features.T).squeeze().item()
        total += cosine_similarity
    return total / bsz


def para_op(theta: float,omega: float) -> float:
    res = 0
    num = len(mask_lists)
    logger.info("%------------------------------------------------%")
    logger.info(f"Theta: {theta}, Omega: {omega}")
    for mask in mask_lists:
        prompt = mask.split("/")[-1].split(".")[0].replace("_1", "")
        draw_img = cv2.imread(mask)
        ref_mask = cv2.imread(ref_mask_list[prompt])
        # imgs here are b*[h,w,c] in lists of numpy
        imgs, info = process(
            prompt,
            draw_img,
            ref_mask=ref_mask,
            seed=seeds[prompt],
            img_count=4,
            start_op_step=st,
            end_op_step=ed,
            loss_alpha=0,
            loss_beta=0,
            step_size=step_size,
            op_step=op_step,
            add_theta=theta,
            add_omega=omega,
        )
        acc = ocr_acc(imgs, info)
        clip_s = clip_score(prompt, imgs)
        logger.info(f"Ocr acc: {acc}, Clip Score: {clip_s}")
        res += 1 - (0.5 * acc + clip_s * 0.5)
    print(res / num)
    logger.info(f"Loss: {res/num}")
    return res / num


parametrization = ng.p.Instrumentation(
    # an integer from 1 to 12
    theta=ng.p.Scalar(lower=-0.5, upper=-0.3),
    omega=ng.p.Scalar(lower=0.6, upper=0.9),
)

optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
recommendation = optimizer.minimize(para_op)
print(recommendation.kwargs)
logger.info(recommendation.kwargs)
