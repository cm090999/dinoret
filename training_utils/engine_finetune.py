import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

from util.customMetrics import DatasetSpecificMetrics

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    
    # switch to train mode
    model.train(True)

    # Check loss function input shape
    expected_loss_input_shape = None
    if args.loss_function == 'CrossEntropyLoss' or args.loss_function == 'LabelSmoothingCrossEntropy' or args.loss_function == 'DistanceWeightedCrossEntropyLoss':
        expected_loss_input_shape = 'Indices'
    elif args.loss_function == 'BCEWithLogitsLoss' or args.loss_function == 'MultiLabelSoftMarginLoss' or args.loss_function == 'SoftTargetCrossEntropy':
        expected_loss_input_shape = 'OneHot'
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    n_samples_per_class = torch.zeros((args.nb_classes), device = device, dtype=int)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        true_label=F.one_hot(targets.to(torch.int64), num_classes=args.nb_classes).type(torch.FloatTensor).cuda()

        # Keep track of number of targets
        histogram = torch.bincount(targets, minlength=args.nb_classes)
        n_samples_per_class += histogram

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if expected_loss_input_shape == 'Indices':
                loss = criterion(outputs, targets)
            elif expected_loss_input_shape == 'OneHot':
                loss = criterion(outputs, true_label)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Training Data Composition: " + str(list(n_samples_per_class.detach().cpu().numpy())))

    return


@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class, criterion, args):

    metrics = DatasetSpecificMetrics(args)

    # Check loss function input shape
    expected_loss_input_shape = None
    if args.loss_function == 'CrossEntropyLoss' or args.loss_function == 'LabelSmoothingCrossEntropy' or args.loss_function == 'DistanceWeightedCrossEntropyLoss':
        expected_loss_input_shape = 'Indices'
    elif args.loss_function == 'BCEWithLogitsLoss' or args.loss_function == 'MultiLabelSoftMarginLoss' or args.loss_function == 'SoftTargetCrossEntropy':
        expected_loss_input_shape = 'OneHot'
    
    metric_logger = misc.MetricLogger(delimiter="  ")

    if mode == 'val':
        header = 'Validation:'
    else:
        header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    rel_image_path_list = []
    abs_image_path_lst = []
    n_samples_per_class = torch.zeros((num_class), device = device, dtype=int)
    prediction_softmax_array = np.zeros((len(data_loader.dataset), num_class))
    true_label_array = np.zeros((len(data_loader.dataset), num_class))
    n_samples_processed = 0
    
    # switch to evaluation mode
    model.eval()

    for idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        curr_batch_size = batch[0].shape[0]
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=args.nb_classes).type(torch.FloatTensor).cuda()

        # Keep track of number of targets
        histogram = torch.bincount(target, minlength=num_class)
        n_samples_per_class += histogram

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if expected_loss_input_shape == 'Indices':
                loss = criterion(output, target)
            elif expected_loss_input_shape == 'OneHot':
                loss = criterion(output, true_label)
            prediction_softmax = nn.Softmax(dim=1)(output)

            prediction_softmax_array[n_samples_processed:n_samples_processed+curr_batch_size,:] = prediction_softmax.cpu().detach().numpy()
            true_label_array[n_samples_processed:n_samples_processed+curr_batch_size,:] = true_label.cpu().detach().numpy()
            rel_image_path_list.extend(batch[2])
            abs_image_path_lst.extend(batch[3])
            n_samples_processed += curr_batch_size

        acc1,acc2 = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)

    # gather the stats from all processes and synchronize
    metric_logger.synchronize_between_processes()

    metrics_dict, results_raw = metrics(true_label_array, prediction_softmax_array, rel_image_path_list, abs_image_path_lst, metric_logger, data_loader, n_samples_per_class, mode)

    return metrics_dict, results_raw

def run_test_evaluation(args,model,device,data_loader_test,criterion,epoch, model_path):
    torch.distributed.barrier()
    #Load best model and evaluate on test set using distributed data parallel
    if not args.eval:
        checkpoint = torch.load(model_path, map_location='cpu')
        best_epoch = checkpoint['epoch']
        print(f"Load pre-trained checkpoint from: {model_path} Best result was at epoch {best_epoch}")
        checkpoint_model = checkpoint['model']
        #add module to the state dict so DDP works
        checkpoint_model = {'module.'+k: v for k, v in checkpoint_model.items()}
        model.load_state_dict(checkpoint_model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    metrics_dict_test, results_raw = evaluate(data_loader_test, model, device,args.task,epoch, mode='test',num_class=args.nb_classes, criterion=criterion, args=args)

    return metrics_dict_test, results_raw
