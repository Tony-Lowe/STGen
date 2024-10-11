import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from t3_dataset import T3DataSet
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil

NUM_NODES = 1
# Configs
batch_size = 3  # default 6
grad_accum = 2  # enable perceptual loss may cost a lot of VRAM, you can set a smaller batch_size and make sure grad_accum * batch_size = 6
ckpt_path = None  # if not None, load ckpt_path and continue training task, will not load "resume_path"
resume_path = "/data/hugging_face_models/cv_anytext_text_generation_editing/anytext_v1.1_art_finetune.ckpt"  # finetune from scratch
model_config = './models_yaml/anytext_sd15.yaml'  # use anytext_sd15_perloss.yaml to enable perceptual loss
logger_freq = 1000
learning_rate = 2e-5  # default 2e-5
mask_ratio = 0  # default 0.5, ratio of mask for inpainting(text editing task), set 0 to disable
wm_thresh = 1.0  # set 0.5 to skip watermark imgs from training(ch:~25%, en:~8%, @Precision93.67%+Recall88.80%), 1.0 not skip
root_dir = './models'  # path for save checkpoints
dataset_percent = 0.0566  # 1.0 use full datasets, 0.0566 use ~200k images for ablation study
save_steps = None  # step frequency of saving checkpoints
save_epochs = 1  # epoch frequency of saving checkpoints
max_epochs = 15  # default 60
assert (save_steps is None) != (save_epochs is None)


if __name__ == '__main__':
    log_img = os.path.join(root_dir, 'image_log/train')
    if os.path.exists(log_img):
        try:
            shutil.rmtree(log_img)
        except OSError:
            pass

    json_paths = [
        r"/data/Datasets/AnyWord-3M/ocr_data/Art/data.json",
        # r'/data/Datasets/AnyWord-3M/ocr_data/COCO_Text/data.json',
        # r'/data/Datasets/AnyWord-3M/ocr_data/icdar2017rctw/data.json',
        # r'/data/Datasets/AnyWord-3M/ocr_data/LSVT/data.json',
        # r'/data/Datasets/AnyWord-3M/ocr_data/mlt2019/data.json',
        # r'/data/Datasets/AnyWord-3M/ocr_data/MTWI2018/data.json',
        # r'/data/Datasets/AnyWord-3M/ocr_data/ReCTS/data.json',
        # '/data/Datasets/AnyWord-3M/laion/data_v1.1.json',
        # '/data/Datasets/AnyWord-3M/wukong_word/wukong_1of5/data_v1.1.json',
        # '/data/Datasets/AnyWord-3M/wukong_word/wukong_2of5/data_v1.1.json',
        # '/data/Datasets/AnyWord-3M/wukong_word/wukong_3of5/data_v1.1.json',
        # '/data/Datasets/AnyWord-3M/wukong_word/wukong_4of5/data_v1.1.json',
        # '/data/Datasets/AnyWord-3M/wukong_word/wukong_5of5/data_v1.1.json',
    ]
    dataset = T3DataSet(json_paths, max_lines=5, max_chars=20, caption_pos_prob=0.0, mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=2, percent=dataset_percent, debug=False, using_dlc=False, wm_thresh=wm_thresh)

    model = create_model(model_config).cpu()
    if ckpt_path is None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = True
    model.only_mid_control = False
    model.unlockKV = False

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=save_steps,
        every_n_epochs=save_epochs,
        save_top_k=3,
        monitor="global_step",
        mode="max",
    )
    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=True, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=-1, precision=32, max_epochs=max_epochs, num_nodes=NUM_NODES, accumulate_grad_batches=grad_accum, callbacks=[logger, checkpoint_callback], default_root_dir=root_dir, strategy='ddp')
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)

    # trainer.save_checkpoint(os.path.join(root_dir, 'anytext_sd15_art_finetune.ckpt'))
