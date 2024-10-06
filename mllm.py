import numpy as np
import requests
import cv2
import json
from openai import OpenAI
import re
from PIL import Image, ImageDraw, ImageFont
from util import draw_pos,draw_glyph_sp

def get_pos_n_glyph(cx,cy,w,h,angle,text,font_path="./font/Arial_Unicode.ttf"):
    rect = [(cx,cy),(w,h),angle]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    positions = draw_pos(box)
    glyphs = draw_glyph_sp(ImageFont.truetype(font_path,size=60),text,rect,scale=2)
    return positions,glyphs

def format_cv2_contour(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx = [j for i in approx.tolist() for j in i]
    # print(approx)
    approx = np.array(approx)
    min_x = np.argmin(approx, axis=0)[0]
    # print(min_x)
    approx = np.concatenate((approx[min_x:], approx[:min_x]))
    # print(approx.tolist())
    len = approx.shape[0]
    # get angles
    angle = []
    for i in range(len):
        vec1 = approx[(i - 1) % len] - approx[i]
        vec2 = approx[(i + 1) % len] - approx[(i) % len]
        angle_rad = np.arccos(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )
        angle_deg = np.rad2deg(angle_rad)
        cross_product = np.cross(vec1, vec2)
        if cross_product < 0:
            angle_deg = 360 - angle_deg
        # print(np.rad2deg((ang1 - ang2) % (2 * np.pi)))
        # angle += [np.rad2deg((ang1 - ang2) % (2 * np.pi))]
        angle = angle + [round(angle_deg, 2)]
    
    return angle, approx.tolist()

def GPT4_maskGen(prompt,key,api_model="gpt-4o"):
    # url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open("template/mask_gen.txt","r") as f:
        template = f.read()
    userprompt = f"Desciption: {prompt} \n Let's think step by step:"
    textprompt = f"{' '.join(template)} \n {userprompt}"
    system_intel = f"{' '.join(template)}"
    client = OpenAI(api_key=api_key)
    print("waiting for GPT-4 response")
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content":system_intel},
            {"role": "user", "content": userprompt}],
    )
    text = response.choices[0].message.content
    # response = requests.request("POST", url, headers=headers, data=payload)
    # obj = response.json()
    # text = obj["choices"][0]["message"]["content"]
    print(text)
    # Extract the split ratio and regional prompt
    return get_json(text)

def GPT4_geometry(texts,polygons,key,api_model="gpt-4o"):
    # url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open("template/Geometry_template.txt","r") as f:
        template = f.read()
    angle, polygons = format_cv2_contour(polygons)
    userprompt = f"Vertices: {polygons} \n Angles: {angle} \n Let's think step by step:"
    textprompt = f"{' '.join(template)} \n {userprompt}"
    system_intel = f"{' '.join(template)}"
    client = OpenAI(api_key=api_key)
    # payload = json.dumps(
    #     {
    #         "model": api_model,  # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details.
    #         "messages": [
    #             {"role": "system", "content": system_intel},
    #             {"role": "user", "content": userprompt},
    #         ],
    #     }
    # )
    # headers = {
    #     "Accept": "application/json",
    #     "Authorization": f"Bearer {api_key}",
    #     "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
    #     "Content-Type": "application/json",
    # }
    print("waiting for GPT-4 response")
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content":system_intel},
            {"role": "user", "content": userprompt}],
    )
    text = response.choices[0].message.content
    # response = requests.request("POST", url, headers=headers, data=payload)
    # obj = response.json()
    # text = obj["choices"][0]["message"]["content"]
    print(text)
    # Extract the split ratio and regional prompt
    return get_json(text)

def GPT4_textsplit(texts,Areas,key,api_model="gpt-4o"):
    # url = "https://api.openai.com/v1/chat/completions"
    api_key = key
    with open("template/Text_split_template.txt","r") as f:
        template = f.read()
    userprompt = f"Areas: {Areas} \n Text: {texts} \n Let's think step by step:"
    textprompt = f"{' '.join(template)} \n {userprompt}"
    system_intel = f"{' '.join(template)}"
    client = OpenAI(api_key=api_key)
    # payload = json.dumps(
    #     {
    #         "model": api_model,  # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details.
    #         "messages": [
    #             {"role": "system", "content": system_intel},
    #             {"role": "user", "content": userprompt},
    #         ],
    #     }
    # )
    # headers = {
    #     "Accept": "application/json",
    #     "Authorization": f"Bearer {api_key}",
    #     "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
    #     "Content-Type": "application/json",
    # }
    print("waiting for GPT-4 response")
    response = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content":system_intel},
            {"role": "user", "content": userprompt}],
    )
    text = response.choices[0].message.content
    # response = requests.request("POST", url, headers=headers, data=payload)
    # obj = response.json()
    # text = obj["choices"][0]["message"]["content"]
    print(text)
    # Extract the split ratio and regional prompt
    return get_json(text)


def get_json(output_text):
    response = output_text
    # Find Final split ratio
    result_string = response.split("```json")[1].split("```")[0]
    # print(result_string)
    res = json.loads(result_string)
    return res


if __name__ == "__main__":
    prompt = "Perspective Conceptive Allusion"

    OPENAI_API_KEY = "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"
    api_model = "gpt-4o"
    # response = GPT4(texts,contours[0],OPENAI_API_KEY,api_model)
    # print(response)
    # for item in response["segmentation"]:
    #     item = item["Vertices"]
    #     res = cv2.drawContours(draw_img, [np.array(item)], -1, (0, 255, 0), 2)
    #     cv2.imshow("region", res)
    #     cv2.waitKey(0)
