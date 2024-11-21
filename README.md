# Beyond Flat Text: Dual Self-inherited Guidance for Visual Text Generation

![sample](docs/paper_teaser.png "sample")


## ⚠️ Attention
We are still organizing our *Divide and Conquer* module and the speed of this code can still be optimized. Our full and optimized code will be released on github upon acceptance.  

## 🛠Installation
```bash
# Install git (skip if already done)
conda install -c anaconda git
# Prepare a font file; Arial Unicode MS is recommended, **you need to download it on your own**
mv your/path/to/arialuni.ttf ./font/Arial_Unicode.ttf
# Create a new environment and install packages as follows:
conda env create -f environment.yaml
conda activate stgen
```

## 🔮Inference
If you have advanced GPU (with at least 8G memory), it is recommended to deploy our demo as below, which includes usage instruction, user interface and abundant examples.
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py
```
FP16 inference is used as default, and a Chinese-to-English translation model is loaded for direct input of Chinese prompt (occupying ~4GB of GPU memory). The default behavior can be modified, as the following command enables FP32 inference and disables the translation model:
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py --use_fp32 --no_translator
```
If FP16 is used and the translation model not used(or load it on CPU, [see here](https://github.com/tyxsspa/AnyText/issues/33)), generation of one single 512x512 image will occupy ~7.5GB of GPU memory.  
In addition, other font file can be used by(although the result may not be optimal):
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py --font_path your/path/to/font/file.ttf
```
You can also load a specified AnyText checkpoint:
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py --model_path your/path/to/your/own/anytext.ckpt
```

In this demo, you can change the style during inference by either change the base model or loading LoRA models(must based on SD1.5):  
- Change base model: Simply fill in your local base model's path in the [Base Model Path].  
- Load LoRA models: Input your LoRA model's path and weight ratio into the [LoRA Path and Ratio]. For example: `/path/of/lora1.pth 0.3 /path/of/lora2.safetensors 0.6`.



