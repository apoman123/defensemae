import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
from torch.distributions.gamma import Gamma

import util.misc as misc
import util.lr_sched as lr_sched
from util.variable_length_padding import padding

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (specs, padding_masks) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            
        # feature transformation
        # samples = transform(wavs)

        # sample a noise with random choosen epsilon by sigma. sigma is from gamma distribution
        # follow the setting from the paper: Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense
        sigma = Gamma(torch.tensor(25), torch.tensor(3))
        noises = torch.randn(specs.shape) * sigma
        
        #print(samples.shape)# 64x3x224x224 for img, 64x1x512x128 for audio
        samples = samples.to(device, non_blocking=True)
        noises = noises.to(device, non_blocking=True)
        padding_masks = padding_masks.to(device, non_blocking=True)
        


        # comment out when not debugging
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # if data_iter_step == 1:
        #     flops = FlopCountAnalysis(model, samples)
        #     print(flops.total())
        #     print(parameter_count_table(model))
        with torch.cuda.amp.autocast():
            if args.finetune_task == "Noisy Spectrogram Modeling" or args.finetune_task == "Masked Noisy Spectrogram Modeling":
                _, pred, _ = model(imgs=samples+noises, mask_ratio=args.mask_ratio, padding_mask=padding_masks)
                reconstructed_specs = model.unpatchify(pred)
                loss_a = torch.mean(reconstructed_specs - samples)
            elif args.finetune_task == "Speech Enhancement":
                _, pred, _ = model(imgs=samples+noises, mask_ratio=args.mask_ratio, padding_mask=padding_masks)
                predicted_noises = model.unpatchify(pred)
                loss_a = torch.mean(noises - predicted_noises)

        loss_value = loss_a.item()
        loss_total = loss_a

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #loss /= accum_iter
        loss_total = loss_total / accum_iter
        loss_scaler(loss_total, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




