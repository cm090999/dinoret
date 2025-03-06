import torch

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.custom_loss import DistanceWeightedCrossEntropyLoss

import warnings 

def generate_training_settings(args, param_groups, num_tasks, dataloader_train):

    # Set mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    if "dinov2" in args.model:
        if (args.block_expansion_positions is not None) and (args.pretrained_checkpoint == ''):

            if args.block_expansion_lr_scaler is not None:
                block_expansion_lr = args.lr * args.block_expansion_lr_scaler
                print(f"Block expansion learning rate is specified. Setting it to {block_expansion_lr}")
            else:
                block_expansion_lr = args.lr
                args.block_expansion_lr_scaler = 1.0
                print(f"Block expansion learning rate is not specified. Setting it to base learning rate: {block_expansion_lr}")
            
            if args.block_expansion_weight_decay is None:
                print(f"Block expansion weight decay is not specified. Setting it to base weight decay: {args.weight_decay}")
                args.block_expansion_weight_decay = args.weight_decay

            param_groups_dict = [{'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.weight_decay},
                    {'params': param_groups[1], 'lr': block_expansion_lr, 'lr_scale': args.block_expansion_lr_scaler, 'weight_decay': args.block_expansion_weight_decay * eff_batch_size / 256}]
            
            optimizer = torch.optim.AdamW(param_groups_dict)
        else: #default case
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay = args.weight_decay)

    elif "RETFound" in args.model:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay = args.weight_decay)
    loss_scaler = NativeScaler()

    loss_weigths = None
    if args.loss_weights == True and args.uniform_class_prob_training == False:
        n_apps_per_class_train = [dataloader_train.dataset.targets.count(i) for i in range(args.nb_classes)]
        class_weights_train = [1.0 / args.nb_classes / n_apps_per_class_train[i] for i in range(len(n_apps_per_class_train))]
        loss_weigths = torch.FloatTensor(class_weights_train).cuda() * len(dataloader_train.dataset)# TODDO: Check scaling

    if args.loss_weights == True and args.uniform_class_prob_training == True:
        warnings.warn("Uniform class probability is activated. Loss weights are not applied!")

    if args.loss_function == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss(weight = loss_weigths, label_smoothing = args.smoothing)
        if mixup_fn is not None:
            mixup_fn = None
            warnings.warn('Mixup is deactivated! Not supported by CrossEntropyLoss!')
    elif args.loss_function == 'DistanceWeightedCrossEntropyLoss':
        criterion = DistanceWeightedCrossEntropyLoss(args.nb_classes, weight_factor=args.loss_weight_factor , penalty_type=args.loss_penalty_type, class_weights=loss_weigths)
        if mixup_fn is not None:
            mixup_fn = None
            warnings.warn('Mixup is deactivated! Not supported by DistanceWeightedCrossEntropyLoss!')
    elif args.loss_function == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss(weight = loss_weigths)
        if args.smoothing > 0.:
            warnings.warn("BCEWithLogitsLoss does not support smoothing!")
    elif args.loss_function == 'MultiLabelSoftMarginLoss':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight = loss_weigths)
        if args.smoothing > 0.:
            warnings.warn("MultiLabelSoftMarginLoss does not support smoothing!")

    # TIMM loss functions
    elif args.loss_function == 'LabelSmoothingCrossEntropy':
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        if mixup_fn is not None:
            mixup_fn = None
            warnings.warn("Mixup is deactivated! Not supported by LabelSmoothingCrossEntropy!")
        if args.loss_weights == True:
            warnings.warn("LabelSmoothingCrossEntropy does not support loss weights!")
    elif args.loss_function == 'SoftTargetCrossEntropy':
        criterion = SoftTargetCrossEntropy()
        if args.loss_weights == True:
            warnings.warn("SoftTargetCrossEntropy does not support loss weights!")
        if args.smoothing > 0.:
            warnings.warn("SoftTargetCrossEntropy does not support smoothing!")
        if mixup_fn is None:
            warnings.warn("Mixup is deactivated!")

    print("criterion = %s" % str(criterion))

    return criterion, optimizer, loss_scaler, mixup_fn