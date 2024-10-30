import os
import argparse

import cv2
import numpy as np
import torch
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, preprocess_image
)

from config import get_config
from models import build_model

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
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
                        default='./output/swinv2_base_patch4_window16_256/ca_ft_3/ckpt_epoch_9.pth')
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
    config.MODEL.NUM_CLASSES = 3
    config.freeze()

    return args, config

def load_checkpoint_easy(config, model):
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    del checkpoint
    torch.cuda.empty_cache()

def de_transform(input_tensor):
    # 将张量转换回NumPy数组，并转换为(x, y, 3)格式
    # 去掉batch维度
    input_tensor = input_tensor.squeeze(0)

    # 将张量从GPU转移到CPU，并转换为NumPy数组
    img_array = input_tensor.cpu().numpy()

    # 逆归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array * std[:, None, None]) + mean[:, None, None]

    # 转换为(x, y, 3)格式
    img_array = np.transpose(img_array, (1, 2, 0))

    img_array = img_array.astype(np.float32) / 255.0

    return img_array

def reshape_transform(tensor, height=8, width=8):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args, config = parse_option()
    device = 'cuda'
    method = "gradcam"
    aug_smooth = False
    eigen_smooth = True
    dataset_name = 'dataset6'
    model_name = 'ours'
    output_dir = f'./cam_vis/{dataset_name}/{model_name}'
    # output_dir = os.path.join(output_dir, 'local')
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        "ablationcam": AblationCAM,
    }

    root_dir = f'/home/pro/DLGNet/GradCAM/{dataset_name}/global'
    root = os.listdir(root_dir)
    image_paths = [os.path.join(root_dir, p) for p in root]
    crop_paths = [p.replace('global', 'local_sam2') for p in image_paths]

    model = build_model(config)
    if hasattr(model, 'flops'):
        flops = model.flops()
    model.cuda()
    load_checkpoint_easy(config, model)
    print(model)

    # for SAM2-SWIN
    # target_layers = [model.layers_g[-1].blocks[-1].norm2]
    target_layers = [model.norm_g]

    # for SwinV2
    # target_layers = [model.layers[-1].blocks[-1]]

    for image_path, crop_path in zip(image_paths, crop_paths):
        global_img = cv2.imread(image_path, 1)[:, :, ::-1]
        local_img = cv2.imread(crop_path, 1)[:, :, ::-1]

        global_img = cv2.resize(global_img, (256, 256))
        input_tensor_global = preprocess_image(global_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(device)
        global_img = np.float32(global_img) / 255.

        local_img = cv2.resize(local_img, (256, 256))
        input_tensor_local = preprocess_image(local_img,
                                               mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]).to(device)
        local_img = np.float32(local_img) / 255.

        input_tensor = [input_tensor_global, input_tensor_local]

        # input_tensor = input_tensor_global

        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        if method == "ablationcam":
            cam = methods[method](model=model,
                                  target_layers=target_layers,
                                  reshape_transform=reshape_transform,
                                  ablation_layer=AblationLayerVit())
        else:
            cam = methods[method](model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        # with cam_algorithm(model=model,
        #                    target_layers=target_layers) as cam:
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
        cam.batch_size = 128
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=aug_smooth,
                            eigen_smooth=eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(global_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # gb_model = GuidedBackpropReLUModel(model=model, device=device)
        # gb = gb_model(input_tensor, target_category=None)
        #
        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)

        cam_output_path = os.path.join(output_dir, f'{os.path.basename(crop_path)[:-4]}.png')

        cv2.imwrite(cam_output_path, cam_image)