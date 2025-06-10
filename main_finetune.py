import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch

import util.misc as misc
from util.tensorboard_utils import log_writer_epoch, init_log_writer, close_log_writer, create_average_metrics, save_metrics_and_images
from util.initialize_training_environment import init_training_env

from model_utils.dataloader_generator import generate_dataloaders
from model_utils.model_generator import generate_model
from training_utils.training_settings_generator import generate_training_settings

from training_utils.engine_finetune import train_one_epoch, evaluate, run_test_evaluation

def get_args_parser():
    parser = argparse.ArgumentParser('fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='dinov2_vitb14', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--forward_patches', default='cls', type=str, choices=['cls', 'patch', 'both'], help='Use CLS or mean of patch tokens or both')
    
    parser.add_argument('--n_classification_heads', default=1, type=int)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    
    parser.add_argument('--preprocessing_CropRoundImage', default = False, type=str2bool)
    parser.add_argument('--preprocessing_CLAHETransform', default = False, type=str2bool)
    parser.add_argument('--preprocessing_MedianFilterTransform', default = False, type=str2bool)

    parser.add_argument('--RandomAdjustSharpness_sharpness_factor', type=float, default = 0.2)
    parser.add_argument('--RandomRotation_degrees', type=float, default = 5)
    parser.add_argument('--Vertical_Horizontal_Flip_Probability', type=float, default = 0.5)

    parser.add_argument('--color_jitter_param_brightness', type=float, default = 0.2, help='(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.06)')
    parser.add_argument('--color_jitter_param_contrast', type=float, default = 0.2, help='(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.06)')
    parser.add_argument('--color_jitter_param_saturation', type=float, default = 0.1, help='(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.06)')
    parser.add_argument('--color_jitter_param_hue', type=float, default = 0.1, help='(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.06)')

    parser.add_argument('--RandomResizeCrop_lowerBound', type=float, default=0.8)
    parser.add_argument('--RandomResizeCrop_upperBound', type=float, default= 1.2)

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=4e-2, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=0.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.0,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--RETFound_pretrained', default='',type=str,
                        help='Location of pretrained RETFound weights')
    parser.add_argument('--task', default='',type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--fix_backbone', default=True, type=str2bool)
    parser.add_argument('--validation_criterion', default='qkappa', type=str, help='Metric used to decide which model is best, higher is better')
    parser.add_argument('--pretrained_checkpoint', default='', type=empty_or_str, help='Path to pretrained checkpoint to evaluate on test set')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/jupyter/Mor_DR_data/data/data/IDRID/Disease_Grading/', type=str,
                        help='dataset path')
    parser.add_argument('--use_filtered_data', default=False, type=str2bool)
    parser.add_argument('--nb_classes', default=5, type=int,
                        help='number of the classification types')
    parser.add_argument('--few_shot_learning', default=-1, type=int,
                        help='If -1, is disabled, else an integer defining the number of training samples per class to use')
    parser.add_argument('--few_shot_learning_seed', default=42, type=int,
                        help='Seed for few shot learning')
    parser.add_argument('--n_few_shot_folds', default=3, type=int, help='Number of few shot training folds')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--uniform_class_prob_training', default=False, type=str2bool)
    parser.add_argument('--uniform_class_prob_validation', default=False, type=str2bool)
    parser.add_argument('--uniform_class_prob_test', default=False, type=str2bool)
    parser.add_argument('--loss_weights', default=True, type=str2bool, help='If True, the loss is weighted by the inverse of the class probability, mixup has to be disabled and uniform_class_prob_training should be False')
    parser.add_argument('--loss_function', default='DistanceWeightedCrossEntropyLoss', choices=['LabelSmoothingCrossEntropy', 'SoftTargetCrossEntropy', 'CrossEntropyLoss', 'BCEWithLogitsLoss', 'MultiLabelSoftMarginLoss', 'DistanceWeightedCrossEntropyLoss'], type=str)
    parser.add_argument('--loss_weight_factor', default=0.1, type=float, help='Weight factor for the distance weighted cross entropy loss')
    parser.add_argument('--loss_penalty_type', default='squared', choices=['linear', 'squared'], type=str, help='Penalty type for the distance weighted cross entropy loss')
    parser.add_argument('--normalize_mean', default=(0.485, 0.456, 0.406))
    parser.add_argument('--normalize_std', default=(0.229, 0.224, 0.225))

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=18, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=int(os.environ['LOCAL_RANK']), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Block expansion parameters
    parser.add_argument('--block_expansion_positions', type=none_or_int, default="None", help='List of positions where to expand the backbone e.g. 1 2 3')
    parser.add_argument('--block_expansion_lr_scaler', default=None, type=float, help='If set, the learning rate of the expanded blocks is multiplied by this factor else the block learning rate is the same as the learning rate')
    parser.add_argument('--block_expansion_weight_decay', default=None, type=float)
    parser.add_argument('--block_expansion_path_dropout', default=0.0, type=float)

    # LoRA adaptation parameters
    parser.add_argument('--lora_adaptation', default=False, type=str2bool, help='If True, LoRA adaptation is applied to the model')
    parser.add_argument('--lora_adaptation_rank', default=8, type=int, help='Rank of LoRA adaptation')
    parser.add_argument('--lora_adaptation_alpha', default=16, type=int, help='Alpha of LoRA adaptation')
    parser.add_argument('--lora_adaptation_target_blocks', type=none_or_int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help='List of blocks to apply LoRA adaptation to')
    parser.add_argument('--lora_adaptation_adapt_attention', default=True, type=str2bool, help='If True, LoRA adaptation is applied to the attention layers')
    parser.add_argument('--lora_adaptation_adapt_mlp', default=True, type=str2bool, help='If True, LoRA adaptation is applied to the MLP layers')

    # Pretraining Parameters
    parser.add_argument('--yaml_config', default=None, type=none_or_str, help='Path to yaml config file')

    return parser

def empty_or_str(value):
    # Convert input to string
    value = str(value)

    # Check if input is 'none'
    if value.lower() == 'none':
        value = ""
    
    return value

def none_or_str(value):
    # Convert input to string
    value = str(value)

    # Check if input is 'none'
    if value.lower() == 'none':
        value = None
    
    return value

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def none_or_int(value):
    print(value)
    print(type(value))
    if 'none' in value.lower():
        return None
    if type(value) == list:
        return value
    if type(value) == str:
        if value.startswith('['):
            # Convert string of shape "[1, 2, 3]" to list of integers
            expansion_list = value.strip('][').split(', ')
            expansion_list = [int(i) for i in expansion_list]

        if value.startswith('"'):
            expansion_list = value.strip('"').split(',') 
            expansion_list = [int(i) for i in expansion_list]

        # Consider if it is a string of shape "1,2,3"
        if value[0].isdigit():

            # Check if separated by comma or space
            if ',' in value:
                expansion_list = value.split(',')

                # Convert list of strings to list of integers
                expansion_list = [int(i) for i in expansion_list]

            if ' ' in value:
                expansion_list = value.split(' ')

                # Convert list of strings to list of integers
                expansion_list = [int(i) for i in expansion_list]

        return expansion_list

def main(args, device, num_tasks, global_rank):

    # Initialize tensorboard writer
    log_writer = init_log_writer(args, global_rank)
    
    # Call all generators
    data_loader_train, data_loader_val, data_loader_test = generate_dataloaders(args = args, num_tasks = num_tasks, global_rank = global_rank)
    model, model_without_ddp, param_groups, n_parameters = generate_model(args = args, device = device)
    criterion, optimizer, loss_scaler, mixup_fn = generate_training_settings(args = args, param_groups = param_groups, num_tasks = num_tasks, dataloader_train = data_loader_train)

    # optionally resume from a checkpoint, overwrites weights and potentially optimizer, etc.
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # If only evaluation is set, run evaluation and exit
    if args.eval:
        metrics_dict_test, results_raw = run_test_evaluation(args,model,device,data_loader_test,criterion,epoch=0, model_path=args.pretrained_checkpoint)
        args.epochs = 0
        epoch = 0

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_metric = -np.inf

    if args.eval == False:
        metrics_dict_test = None

    for epoch in range(args.start_epoch, args.epochs):

        # Manage Sampler in distributed training
        if args.distributed and not args.uniform_class_prob_training:
            data_loader_train.sampler.set_epoch(epoch)

        # Train one epoch
        train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
            )

        # Run validation
        metrics_dict_val, _ = evaluate(data_loader_val, model, device,args.task,epoch, mode='val',num_class=args.nb_classes, criterion=criterion, args=args)
        ## Check if best model
        if best_metric < metrics_dict_val[args.validation_criterion]:
            print(f"New best model at epoch {epoch} with {args.validation_criterion} {metrics_dict_val[args.validation_criterion]}")
            best_metric = metrics_dict_val[args.validation_criterion]
            
            if args.task:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        
        # TEST best model after last epoch
        if epoch==(args.epochs-1):
            metrics_dict_test, results_raw = run_test_evaluation(args,model,device,data_loader_test,criterion,epoch, model_path=os.path.join(args.task,'checkpoint-best.pth'))

        # Log metrics
        log_writer_epoch(log_writer, epoch, metrics_dict_val)

    # Save all metrics
    save_metrics_and_images(log_writer, args, results_raw, metrics_dict_test, data_loader_train, data_loader_test)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Close Log Writer
    close_log_writer(log_writer)

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()

    # Add yaml config to args if provided
    if args.yaml_config:
        import yaml
        # Load yaml config
        with open(args.yaml_config, 'r') as file:
            yaml_args = yaml.load(file, Loader=yaml.FullLoader)

        # # Add yaml args to args with prefix 'yaml_'
        # for key, value in yaml_args.items():
        #     setattr(args, f'yaml_{key}', value)
        def assign_nested_attrs(dest_obj, source_dict, prefix=""):
            for key, value in source_dict.items():
                full_key = f"{prefix}_{key}".strip("_")
                if isinstance(value, dict):
                    # Handle nested levels
                    # setattr(dest_obj, full_key, type(dest_obj)())  # Create a nested object
                    assign_nested_attrs(dest_obj, value, prefix=full_key)
                else:
                    setattr(dest_obj, full_key, value)

        assign_nested_attrs(args, yaml_args, prefix="yaml")

    if args.block_expansion_positions == [None]:
        args.block_expansion_positions = None

    print(args.block_expansion_positions)

    if args.task:
        Path(args.task).mkdir(parents=True, exist_ok=True)

    # Intialize training environment (Set seed, distributed mode, etc.)
    device, num_tasks, global_rank = init_training_env(args)

    if args.few_shot_learning == -1:
        main(args, device, num_tasks, global_rank)

    else:
        task = args.task
        for few_shot in range(args.n_few_shot_folds):

            args.task = task + f'/few_shot_{few_shot}'
            if args.task:
                Path(args.task).mkdir(parents=True, exist_ok=True)
            args.few_shot_learning_seed += 1
            
            main(args, device, num_tasks, global_rank)

        args.task = task
        create_average_metrics(args, global_rank)
