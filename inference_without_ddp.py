import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from sklearn.metrics import classification_report
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from models import build_model

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
                        # default='./configs/base.yaml')
                        default='./configs/dynamic.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint',
                        # default='./output/swinv2_base_patch4_window16_256/swin_base_v2_g/ckpt_epoch_30.pth')
                        default='./output/swinv2_base_patch4_window16_256/ca_ft_baseline/ckpt_epoch_9.pth')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_false', help='Perform evaluation only')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_false',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_false', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    config.defrost()
    # for segment evaluation
    config.DATA.DATA_PATH = r'/home/pro/datasets/segment/dataset1/global/train'
    config.MODEL.NUM_CLASSES = 3
    config.AMP_ENABLE = False
    config.DATA.CENTER = "dataset1_seg"
    config.RETURN_ATTN = True
    config.RETURN_XFEATURE = False
    config.freeze()

    return args, config


def load_checkpoint_easy(config, model):
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    del checkpoint
    torch.cuda.empty_cache()

def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class_labels = ['normal', 'benign', 'tumor']

def make_dataset(path, is_train=True):
    imgs = []
    for i in range(len(class_labels)):
        if i == 0:
            continue
        img_dir_path = os.path.join(path, class_labels[i])
        img_name = os.listdir(img_dir_path)
        for name in img_name:
            img_path = os.path.join(img_dir_path, name)
            local_path = img_path.replace('global', 'local_sam2')
            # local_path = img_path.replace('global', 'maskrcnn')
            # local_path = img_path.replace('global', 'unet')
            imgs.append((img_path, local_path, i, name))

    return imgs

class LPCDataset(data.Dataset):
    def __init__(self, root, transform=None, is_train=True):
        self.imgs = make_dataset(root, is_train)
        self.transform = transform

    def __getitem__(self, index):
        img_path, local_path, label, name = self.imgs[index]

        img_x = Image.open(img_path)
        local_x = Image.open(local_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
            local_x = self.transform(local_x)

        return [img_x, local_x], label, name[:-len('.png')]

    def __len__(self):
        return len(self.imgs)

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'lpc':
        if is_train:
            data_dir = config.DATA.DATA_PATH
        else:
            data_dir = config.DATA.DATA_PATH.replace('train', 'test') if config.EVAL_MODE else (
                config.DATA.DATA_PATH.replace('train', 'val'))
            data_dir = data_dir.replace('dataset1', config.DATA.CENTER)
        dataset = LPCDataset(data_dir, transform, is_train)
        nb_classes = 3
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes

@torch.no_grad()
def inference(config, data_loader, model):
    model.eval()

    all_targets = []
    all_predictions = []
    all_scores = []
    all_names = []
    if config.RETURN_FEATURE:
        all_features = []
        tns_save_path = r"/home/pro/current/swin_transformer/tsne"
        if not os.path.exists(tns_save_path):
            os.makedirs(tns_save_path)
    if config.RETURN_ATTN:
        num_layers = len(config.MODEL.SWINV2.DEPTHS)
        all_attns = []
        attn_save_path = r"/home/pro/current/swin_transformer/attn/dataset1_seg"
        if not os.path.exists(attn_save_path):
            os.makedirs(attn_save_path)
    if config.RETURN_XFEATURE:
        num_layers = len(config.MODEL.SWINV2.DEPTHS)
        all_feats_before = []
        all_feats_after = []
        feats_save_path = r"/home/pro/current/swin_transformer/xfeature/dataset1"
        if not os.path.exists(feats_save_path):
            os.makedirs(feats_save_path)

    for idx, (images, target, name) in tqdm(enumerate(data_loader)):
        images = [image.cuda(non_blocking=True) for image in images]
        target = target.cuda(non_blocking=True)
        all_names.extend(name)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            if config.RETURN_FEATURE:
                features, output = model(images, return_features=True)
                all_features.append(features.cpu().numpy())
            elif config.RETURN_ATTN:
                attn_weights, output = model(images, return_attn=True)
                # 遍历每层的注意力权重
                for layer_idx, layer_attn in enumerate(attn_weights):
                    # layer_attn 是每一层的注意力权重
                    # 将每个头的权重转换为 NumPy 数组并添加到对应层的列表中
                    attn_heads = [head_attn.cpu().numpy() for head_attn in layer_attn]
                    all_attns.append(attn_heads)  # 保存每层的头权重
            elif config.RETURN_XFEATURE:
                features_before, features_after, output = model(images, return_xfeatures=True)
                for layer_idx, layer_feat in enumerate(features_before):
                    all_feats_before.append(layer_feat.cpu().numpy())

                for layer_idx, layer_feat in enumerate(features_after):
                    all_feats_after.append(layer_feat.cpu().numpy())

            else:
                output = model(images)
        if isinstance(output, list):
            output = output[-1]

        # output = adjust_logits_with_temperature(output, 0.3)
        # Apply softmax to get probabilities for each class
        output_probs = F.softmax(output, dim=1)
        # output_probs = output

        # Store the scores (probabilities)
        all_scores.extend(output_probs.cpu().numpy())

        _, preds = torch.max(output, 1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())

    if config.RETURN_ATTN:
        for layer_idx in range(num_layers):
            attn_array = np.concatenate(all_attns[layer_idx::num_layers], axis=0)
            # attn_array = np.array(all_attns[layer_idx])  # 转换为 NumPy 数组
            for i, name in enumerate(all_names):
                np.save(os.path.join(attn_save_path, f"{name}_attn_layer{layer_idx}.npy"), attn_array[i])

    if config.RETURN_XFEATURE:
        for layer_idx in range(num_layers):
            feats_before = np.concatenate(all_feats_before[layer_idx::num_layers], axis=0)
            feats_after = np.concatenate(all_feats_after[layer_idx::num_layers], axis=0)
            for i, name in enumerate(all_names):
                np.save(os.path.join(feats_save_path, f"{name}_layer{layer_idx}_before.npy"), feats_before[i])
                np.save(os.path.join(feats_save_path, f"{name}_layer{layer_idx}_after.npy"), feats_after[i])

        print(f"Features saved to {feats_save_path}")

    classes = {
        0: 'normal',
        1: 'benign',
        2: 'tumor'
    }
    preds_save = [classes[x] for x in all_predictions]
    targets_save = [classes[x] for x in all_targets]
    results = pd.DataFrame({'names': all_names, 'preds': preds_save, 'targets': targets_save})
    # for segment
    # results.to_csv(os.path.join(r"/home/pro/SAM_adapter/MedSAM2/work_dir/MedSAM2-Tiny-prompt-frozen/outputs",
    #                             "results_cls.csv"), index=False, header=True)
    # results.to_csv(os.path.join(r"/home/pro/SAM/exp4", "results_cls.csv"), index=False, header=True)
    # results.to_csv(os.path.join(r"/home/pro/current/Pytorch-UNet", "results_cls.csv"), index=False, header=True)
    # for i in range(3):
    #     results[f'class_{i}_score'] = [output_scores[i] for output_scores in all_scores]
    # results.to_csv(os.path.join(r"/home/pro/DLGNet/roc/ours/", "results.csv"), index=False, header=True)

    target_names = ['Normal', 'Benign', 'Tumor']
    # Compute confusion matrix
    # cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2])
    report = classification_report(all_targets, all_predictions, target_names=target_names, digits=4)
    print(report)


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    model = build_model(config)
    if hasattr(model, 'flops'):
        flops = model.flops()
    model.cuda()
    load_checkpoint_easy(config, model)

    dataset_val, _ = build_dataset(is_train=False, config=config)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    inference(config, data_loader_val, model)