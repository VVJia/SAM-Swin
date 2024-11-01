# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import os
import argparse
import datetime
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
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
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        if not config.EVAL_MODE:
            acc1, loss, recalls = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            # return
        else:
            inference(config, data_loader_val, model)
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, loss, recalls = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1) or
                                     epoch >= config.TRAIN.EPOCHS - 10):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger, tag=None)

        acc1, loss, recalls = validate(config, data_loader_val, model)
        metrecs = 0.5 * acc1 + (recalls[0] * 0.5 + recalls[1] * 0.5) * 0.5
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {metrecs:.1f}%")
        if metrecs > max_accuracy:
            save_checkpoint(config, epoch, model_without_ddp, metrecs, optimizer, lr_scheduler, loss_scaler,
                            logger, tag="best")
        max_accuracy = max(max_accuracy, metrecs)
        logger.info(f'Max accuracy/recall: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    losses_meter = [AverageMeter() for _ in range(len(config.MODEL.LOSS_WEIGHT))]

    start = time.time()
    end = time.time()

    for idx, (samples, targets, _) in enumerate(data_loader):
        samples = [sample.cuda(non_blocking=True) for sample in samples]
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(samples)
        if isinstance(outputs, list):
            losses = [criterion(output, targets) for output in outputs]

            if config.MODEL.LOSS_WEIGHT is not None and config.MODEL.DEEP_SUPERVISION:
                losses = [losses[i] + losses[i + 1] for i in range(0, 8, 2)] + [losses[8]]
                losses_weight = config.MODEL.LOSS_WEIGHT
                loss = sum([l*w for l, w in zip(losses, losses_weight)])
            else:
                loss = sum(losses)
        else:
            loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if losses:
            for i in range(len(losses_meter)):
                losses_meter[i].update(losses[i].item() / config.TRAIN.ACCUMULATION_STEPS, targets.size(0))

        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                # f'loss_0 {losses_meter[0].val:.4f} ({losses_meter[0].avg:.4f})\t'
                # f'loss_1 {losses_meter[1].val:.4f} ({losses_meter[1].avg:.4f})\t'
                # f'loss_2 {losses_meter[2].val:.4f} ({losses_meter[2].avg:.4f})\t'
                # f'loss_3 {losses_meter[3].val:.4f} ({losses_meter[3].avg:.4f})\t'
                # f'loss_weghts {losses_weight}\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    losses_meter = [AverageMeter() for _ in range(len(config.MODEL.LOSS_WEIGHT))]

    # Initialize AverageMeter instances for class-specific recall
    recall1_meter = AverageMeter()  # Recall for class 1
    recall2_meter = AverageMeter()  # Recall for class 2

    end = time.time()
    for idx, (images, targets, _) in enumerate(data_loader):
        images = [image.cuda(non_blocking=True) for image in images]
        targets = targets.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(images)
        if isinstance(outputs, list):
            losses = [criterion(output, targets) for output in outputs]

            if config.MODEL.LOSS_WEIGHT is not None and config.MODEL.DEEP_SUPERVISION:
                losses = [losses[i] + losses[i + 1] for i in range(0, 8, 2)] + [losses[8]]
                loss = sum([l * w for l, w in zip(losses, config.MODEL.LOSS_WEIGHT)])
            else:
                loss = sum(losses)
            outputs = outputs[-1]
        else:
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1 = accuracy(outputs, targets, topk=(1, ))[0]
        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)

        if losses:
            losses_ = [reduce_tensor(t) for t in losses]

        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))

        if losses:
            for i in range(len(losses_meter)):
                losses_meter[i].update(losses_[i].item(), targets.size(0))

        # Get predicted labels
        _, preds = torch.max(outputs, dim=1)

        # Calculate recall for class 1 and class 2 locally
        relevant_1 = (targets == 1).sum().float()
        true_positives_1 = ((preds == 1) & (targets == 1)).sum().float()
        relevant_2 = (targets == 2).sum().float()
        true_positives_2 = ((preds == 2) & (targets == 2)).sum().float()

        # Use all_reduce to aggregate across all GPUs
        relevant_1 = reduce_tensor(relevant_1)
        relevant_2 = reduce_tensor(relevant_2)
        true_positives_1 = reduce_tensor(true_positives_1)
        true_positives_2 = reduce_tensor(true_positives_2)

        # Calculate recall only after the reduction
        if relevant_1 > 0:  # Avoid division by zero
            recall1 = true_positives_1 / relevant_1 * 100.0
            recall1_meter.update(recall1.item(), relevant_1.item())

        if relevant_2 > 0:  # Avoid division by zero
            recall2 = true_positives_2 / relevant_2 * 100.0
            recall2_meter.update(recall2.item(), relevant_2.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                # f'loss_0 {losses_meter[0].val:.4f} ({losses_meter[0].avg:.4f})\t'
                # f'loss_1 {losses_meter[1].val:.4f} ({losses_meter[1].avg:.4f})\t'
                # f'loss_2 {losses_meter[2].val:.4f} ({losses_meter[2].avg:.4f})\t'
                # f'loss_3 {losses_meter[3].val:.4f} ({losses_meter[3].avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Recall@1 {recall1_meter.val:.3f} ({recall1_meter.avg:.3f})\t'
                f'Recall@2 {recall2_meter.val:.3f} ({recall2_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
    logger.info(f' * Recall@1 {recall1_meter.avg:.3f}')
    logger.info(f' * Recall@2 {recall2_meter.avg:.3f}')
    return acc1_meter.avg, loss_meter.avg, [recall1_meter.avg, recall2_meter.avg]


@torch.no_grad()
def inference(config, data_loader, model):
    model.eval()

    batch_time = AverageMeter()
    acc1_meter = AverageMeter()

    all_targets = []
    all_predictions = []
    all_scores = []
    all_names = []

    end = time.time()
    for idx, (images, target, name) in enumerate(data_loader):
        images = [image.cuda(non_blocking=True) for image in images]
        target = target.cuda(non_blocking=True)
        all_names.extend(name)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        if isinstance(output, list):
            output = output[-1]

        # Apply softmax to get probabilities for each class
        output_probs = F.softmax(output, dim=1)

        # Store the scores (probabilities)
        all_scores.extend(output_probs.cpu().numpy())

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1, ))[0]
        acc1 = reduce_tensor(acc1)

        acc1_meter.update(acc1.item(), target.size(0))

        _, preds = torch.max(output, 1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')


    # Gather all targets, predictions, and scores from all GPUs
    all_targets_gathered = [None for _ in range(dist.get_world_size())]
    all_predictions_gathered = [None for _ in range(dist.get_world_size())]
    all_scores_gathered = [None for _ in range(dist.get_world_size())]
    all_names_gathered = [None for _ in range(dist.get_world_size())]

    # Use dist.all_gather to gather data from all processes
    dist.all_gather_object(all_targets_gathered, all_targets)
    dist.all_gather_object(all_predictions_gathered, all_predictions)
    dist.all_gather_object(all_scores_gathered, all_scores)
    dist.all_gather_object(all_names_gathered, all_names)

    # Flatten the lists from all GPUs into one
    all_targets = [item for sublist in all_targets_gathered for item in sublist]
    all_predictions = [item for sublist in all_predictions_gathered for item in sublist]
    all_scores = [item for sublist in all_scores_gathered for item in sublist]
    all_names = [item for sublist in all_names_gathered for item in sublist]

    classes = {
        0: 'normal',
        1: 'benign',
        2: 'tumor'
    }
    preds_save = [classes[x] for x in all_predictions]
    targets_save = [classes[x] for x in all_targets]
    results = pd.DataFrame({'names': all_names, 'preds': preds_save, 'targets': targets_save})
    for i in range(3):
        results[f'class_{i}_score'] = [output_scores[i] for output_scores in all_scores]
    results.to_csv(os.path.join(config.OUTPUT, "results.csv"), index=False, header=True)

    target_names = ['Normal', 'Benign', 'Tumor']
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2])
    report = classification_report(all_targets, all_predictions, target_names=target_names, digits=4)

    # Compute metrics
    recall = cm.diagonal() / cm.sum(axis=1)  # Recall for each class
    precision = cm.diagonal() / cm.sum(axis=0)  # Precision for each class
    specificity = []
    for i in range(len(cm)):
        true_negatives = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        specificity.append((true_negatives / (true_negatives + cm[:, i].sum() - cm[i, i])).item())

    logger.info(f' * Acc@1 {acc1_meter.avg:.2f}')
    logger.info(f' * Recall: Normal {recall[0]*100:.2f}, Benign {recall[1]*100:.2f}, Tumor {recall[2]*100:.2f}')
    logger.info(f' * Precision: Normal {precision[0]*100:.2f}, Benign {precision[1]*100:.2f}, Tumor {precision[2]*100:.2f}')
    logger.info(f' * Specificity: Normal {specificity[0]*100:.2f}, Benign {specificity[1]*100:.2f}, Tumor {specificity[2]*100:.2f}')
    logger.info(f'\nClassification Report:\n{report}')


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = [image.cuda(non_blocking=True) for image in images]
        batch_size = images[0].shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
