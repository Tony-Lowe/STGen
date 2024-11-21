# Beyond Flat Text: Dual Self-inherited Guidance for Visual Text Generation

![sample](docs/paper_teaser.png "sample")

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

## ⚠️ Attention
We are still organizing our *Divide and Conquer* module and the speed of this code can still be optimized. Our full and optimized code will be released on github upon acceptance.  

