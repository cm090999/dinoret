import json
import random

import torch

from util.datasets import get_datasets
import util.misc as misc

def create_index_lists(dataset):
    num_classes = max(dataset.targets) + 1

    class_indices = [[index for index, element in enumerate(dataset.targets) if element == i] for i in range(num_classes)]

    return class_indices

def generate_dataloaders(args, num_tasks, global_rank):

    if ":" in args.data_path:
        multi_dataset_list = args.data_path.split(":")
        training_datasets =[]
        validation_datasets = []
        test_datasets = []
        modifed_args = args
        for path in multi_dataset_list:
            modifed_args.data_path = path
            dataset_train, dataset_val, dataset_test = get_datasets(modifed_args)
            training_datasets.append(dataset_train)
            validation_datasets.append(dataset_val)
            test_datasets.append(dataset_test)
        dataset_train = torch.utils.data.ConcatDataset(training_datasets)
        dataset_val = torch.utils.data.ConcatDataset(validation_datasets)
        dataset_test = torch.utils.data.ConcatDataset(test_datasets)

        dataset_train.targets = [target for dataset in training_datasets for target in dataset.targets]
        dataset_val.targets = [target for dataset in validation_datasets for target in dataset.targets]
        dataset_test.targets = [target for dataset in test_datasets for target in dataset.targets]
        
        dataset_train.classes = training_datasets[0].classes
        dataset_val.classes = validation_datasets[0].classes
        dataset_test.classes = test_datasets[0].classes

        print(f"Combined Training dataset: {len(dataset_train)} samples")
        print(f"Combined Validation dataset: {len(dataset_val)} samples")
        print(f"Combined Test dataset: {len(dataset_test)} samples")

    else:
        dataset_train, dataset_val, dataset_test = get_datasets(args)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        
    if args.uniform_class_prob_training == True:
        n_apps_per_class_train = [dataset_train.targets.count(i) for i in range(args.nb_classes)]
        class_weights_train = [1.0 / args.nb_classes / n_apps_per_class_train[i] for i in range(len(n_apps_per_class_train))]
        sample_weights_train = [class_weights_train[dataset_train.targets[i]] for i in range(len(dataset_train))]
        uniform_sampler_train = torch.utils.data.WeightedRandomSampler(weights=sample_weights_train, num_samples=len(dataset_train), replacement=True)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=uniform_sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)


    if args.uniform_class_prob_validation == True:
        n_apps_per_class_val = [dataset_val.targets.count(i) for i in range(args.nb_classes)]
        class_weights_val = [1.0 / args.nb_classes / n_apps_per_class_val[i] for i in range(len(n_apps_per_class_val))]
        sample_weights_val = [class_weights_val[dataset_val.targets[i]] for i in range(len(dataset_val))]
        uniform_sampler_val = torch.utils.data.WeightedRandomSampler(weights=sample_weights_val, num_samples=len(dataset_val), replacement=True)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=uniform_sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)

    else:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)
        
    if args.uniform_class_prob_test == True:
        n_apps_per_class_test = [dataset_test.targets.count(i) for i in range(args.nb_classes)]
        class_weights_test = [1.0 / args.nb_classes / n_apps_per_class_test[i] for i in range(len(n_apps_per_class_test))]
        sample_weights_test = [class_weights_test[dataset_test.targets[i]] for i in range(len(dataset_test))]
        uniform_sampler_test = torch.utils.data.WeightedRandomSampler(weights=sample_weights_test, num_samples=len(dataset_test), replacement=True)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=uniform_sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)
        
    else:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)

    return data_loader_train, data_loader_val, data_loader_test