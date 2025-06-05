import os
import json
import util.misc as misc
import pandas as pd
import ast
import random
import matplotlib.pyplot as plt
import matplotlib
import re
from PIL import Image
from sklearn.metrics import confusion_matrix
from pycm import ConfusionMatrix
import itertools
from textwrap import wrap
import warnings

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tbparse import SummaryReader

def is_main_process():
    if dist.is_available():
        return dist.get_rank() == 0
    else:
        return True

# Initialize tensorboard writer
def init_log_writer(args, global_rank):

    if global_rank == 0 and args.task is not None:
        os.makedirs(args.task, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.task,'tensorboard_data'))
    else:
        log_writer = None

    return log_writer

def close_log_writer(log_writer):
    if is_main_process():
        if log_writer is not None:
            log_writer.close()
    return

#function to save hparams to tensorboard
def save_hparams(writer, args, metrics = None):
    #Only save in main process
    if is_main_process(): 
        hparams = vars(args)
        #convert classification report to dict
        classification_report = metrics['classification_report']
        classification_report_list, names = reorganize_dict(classification_report)
        for metric_dict, name in zip(classification_report_list, names):
            for label, value in metric_dict.items():
                metrics[f'test_report/{name}/{label}'] = value

        #convert all lists to strings
        hparams = check_dict(hparams)
        metrics = check_dict(metrics)

        assert isinstance(hparams, dict), "args must be a dict"
        assert isinstance(metrics, dict), "metrics must be a dict"

        writer.add_hparams(hparams, metrics)

#Remove all lists from dict otherwise tensorboard will throw an error
def check_dict(dict):
    new_dict = {}
    for key in dict:
        if isinstance(dict[key], int) or isinstance(dict[key], float) or isinstance(dict[key], str) or isinstance(dict[key], bool):
            new_dict[key] = dict[key]
    return new_dict

def save_classification_report(writer, metrics, epoch, prefix=''):
    # Only save in main process
    if dist.get_rank() == 0:
        metrics_list, metric_names = reorganize_dict(metrics)
 
        for metric, name in zip(metrics_list, metric_names):
            assert isinstance(metric, dict), "metrics must be a dict"
            
            for label, value in metric.items():
                writer.add_scalar(f'{prefix}/{name}/{label}', value, epoch)

def reorganize_dict(data):
    precision_dict = {}
    recall_dict = {}
    f1_score_dict = {}
    support_dict = {}

    for label, metrics in data.items():
        if isinstance(metrics, dict):
            if 'precision' in metrics:
                precision_dict[label] = metrics['precision']
            if 'recall' in metrics:
                recall_dict[label] = metrics['recall']
            if 'f1-score' in metrics:
                f1_score_dict[label] = metrics['f1-score']
            if 'support' in metrics:
                support_dict[label] = metrics['support']

    return [precision_dict, recall_dict, f1_score_dict, support_dict], ['precision', 'recall', 'f1-score', 'support']


def log_epoch_values(log_writer, epoch, metrics_dict, prefix=''):
    #save metrics to tensorboard with 
    if log_writer is not None:

        for key in metrics_dict:
            if isinstance(metrics_dict[key], list) or isinstance(metrics_dict[key], dict):
                pass #skip lists and dicts
            else:
                log_writer.add_scalar(f"{prefix}/{key}", metrics_dict[key], epoch)

        log_writer.add_histogram('sample_distribution', np.array(metrics_dict['n_samples_per_class']), epoch)
        save_classification_report(log_writer, metrics_dict['classification_report'], epoch, prefix=f"{prefix}_report")

    return

def log_writer_epoch(log_writer, epoch, metrics_dict_val):

    # Log val metrics to tensorboard
    log_epoch_values(log_writer, epoch, metrics_dict_val, prefix='val')

    return

def save_arguments_to_file(args, filename="arguments.json"):
    with open(os.path.join(args.task, filename), "w") as file:
        json.dump(vars(args), file, indent=4)

# Define a function to convert strings to lists
def convert_to_list(element):
    if isinstance(element, str):
        return ast.literal_eval(element)
    else:
        return element

def save_all_metrics(args, metrics_dict_test, log_writer, results_raw):

    with open(os.path.join(args.task, 'metrics_test.txt'), 'w') as txt_file:
        json.dump(metrics_dict_test, txt_file, indent=4)
    with open(os.path.join(args.task, 'metrics_test.json'), 'w') as json_file:
        json.dump(metrics_dict_test, json_file)
                
    # Save arguments to a file
    save_arguments_to_file(args)
    save_hparams(log_writer, args, metrics_dict_test)

    #save numpy array in csv file
    test_predictions = pd.DataFrame(results_raw)

    # Save test_predictions to csv
    test_predictions.to_csv(os.path.join(args.task,"test_predictions.csv"), index=False)

    return

def get_all_run_files(run_folder, recursion_depth=0):
    max_recursion_depth = 1
    if recursion_depth > max_recursion_depth:
        return []
    
    file_list = []
    for file_name in os.listdir(run_folder):
        if os.path.isdir(os.path.join(run_folder, file_name)):
            file_list += (get_all_run_files(os.path.join(run_folder, file_name), recursion_depth+1))
        elif 'events.out.tfevents' in file_name:
            file_list.append(os.path.join(run_folder, file_name))
    return file_list

def load_run_data(run_folder):
    run_files = get_all_run_files(run_folder)
    if len(run_files) != 2:
        raise ValueError(f"Not correct number of files found. Expected 2 got {len(run_files)}")


    try:
        
        #val and train file
        reader_val = SummaryReader(run_files[0])
        val_data = pd.DataFrame(reader_val.scalars)

        #test file with hparams and test results
        reader_test = SummaryReader(run_files[1])
        test_results =  pd.DataFrame(reader_test.scalars)
        hparams = pd.DataFrame(reader_test.hparams)
        #check if any dataframe is empty
        if val_data.empty or test_results.empty or hparams.empty:
            raise ValueError("One of the dataframes is empty")
        
    except:
        #val and train file
        reader_val = SummaryReader(run_files[1])
        val_data = pd.DataFrame(reader_val.scalars)

        #test file with hparams and test results
        reader_test = SummaryReader(run_files[0])
        test_results =  pd.DataFrame(reader_test.scalars)
        hparams = pd.DataFrame(reader_test.hparams)
        #check if any dataframe is empty

        if val_data.empty or test_results.empty or hparams.empty:
            if hparams.empty:
                print("hparams is empty")
            raise ValueError("One of the dataframes is empty")

    return val_data, hparams, test_results

def find_event_files(base_folder):
    event_files = []
    for run_folder in os.listdir(base_folder):
        run_path = os.path.join(base_folder, run_folder, 'tensorboard_data')
        if os.path.isdir(run_path):
            for file in os.listdir(run_path):
                if 'events.out.tfevents' in file:
                    event_files.append(run_path)
                    break
    return event_files

def make_mean(base_folder):

    run_folders = find_event_files(base_folder)

    # Get data from runs
    val_data_list, hparams_list, test_data_list = [0] * len(run_folders), [0] * len(run_folders), [0] * len(run_folders)
    for i,run_folder in enumerate(run_folders):
        val_data_list[i], hparams_list[i], test_data_list[i] = load_run_data(run_folder)

    ## Make average of all runs
    # Create copy of dataframes
    test_data_avg = test_data_list[0].copy()
    val_data_avg = val_data_list[0].copy()
    
    # Set entries to 0
    test_data_avg['value'] = 0
    val_data_avg['value'] = 0

    # Add all values
    for i in range(len(run_folders)):
        test_data_avg['value'] += test_data_list[i]['value']
        val_data_avg['value'] += val_data_list[i]['value']

    # Divide by number of runs
    test_data_avg['value'] /= len(run_folders)
    val_data_avg['value'] /= len(run_folders)

    return val_data_avg, test_data_avg, hparams_list[0]

def save_dataframe_to_tensorboard(writer, val_data, test_data, hparams):

    # Convert hparams to dict
    hparams_dict = {}
    for index, row in hparams.iterrows():
        hparams_dict[row['tag']] = row['value']

    # Convert test_data to dict
    test_data_dict = {}
    for index, row in test_data.iterrows():
        test_data_dict[row['tag']] = row['value']

    assert isinstance(hparams_dict, dict), "args must be a dict"
    assert isinstance(test_data_dict, dict), "metrics must be a dict"

    writer.add_hparams(hparams_dict, test_data_dict)

    for _,row in val_data.iterrows():

        writer.add_scalar(row['tag'], row['value'], row['step'])

    return

def denormalize(image, mean, std):

    # Convert image to floating point to prevent integer overflow
    image = image.astype(np.float32)
    
    # Apply denormalization in a vectorized manner
    image = (image * std) + mean

    return image

def clamp_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image

def make_plot_of_wrong_image(augemnted_image, image_id, prediction, gt, predictions, args):

    # Get image
    img = Image.open(image_id)

    # Create figure
    fig , ax = plt.subplots(1,3, figsize=(30, 10), tight_layout=True)

    # Plot original image
    ax[0].imshow(img)
    ax[0].set_aspect('auto')

    # Plot augmented image
    augemnted_image = np.transpose(augemnted_image.numpy(), (1, 2, 0))
    augemnted_image = denormalize(augemnted_image, args.normalize_mean, args.normalize_std)
    augemnted_image = clamp_image(augemnted_image)
    ax[1].imshow(augemnted_image)
    ax[1].set_aspect('auto')

    # Plot bar plot of predicitons
    bar_colors = ['blue'] * len(predictions)  # Default color for all bars
    bar_colors[gt] = 'red'
    ax[2].bar(np.arange(args.nb_classes), predictions, color = bar_colors)

    # Add image titles
    ax[0].set_title('Original image', fontsize=30)
    ax[1].set_title('Augmented image', fontsize=30)
    ax[2].set_title('Predicted Probabilities', fontsize=30)

    # Add figure title
    fig.suptitle(f'Prediction: {prediction}, Ground Truth: {gt}', fontsize=40)

    # Remove coordinates
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    # Close figure
    plt.close(fig)

    return fig

def make_plot_of_augmentations(dataloader_sample, args):

    # Get image, label and original image path
    augemnted_image = dataloader_sample[0]
    label = dataloader_sample[1]
    abs_path = dataloader_sample[3]

    # Get original image
    image_orig = Image.open(abs_path)

    # Create figure
    fig , ax = plt.subplots(1,2, figsize=(20, 10), tight_layout=True)

    # Plot original image
    ax[0].imshow(image_orig)
    ax[0].set_aspect('auto')

    # Plot augmented image
    augemnted_image = np.transpose(augemnted_image.numpy(), (1, 2, 0))
    augemnted_image = denormalize(augemnted_image, args.normalize_mean, args.normalize_std)
    augemnted_image = clamp_image(augemnted_image)
    ax[1].imshow(augemnted_image)
    ax[1].set_aspect('auto')

    # Add image titles
    ax[0].set_title('Original image', fontsize=30)
    ax[1].set_title('Augmented image', fontsize=30)

    # Add figure title
    fig.suptitle(f'Ground Truth: {label}', fontsize=40)

    # Remove coordinates
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    # Close figure
    plt.close(fig)

    return fig

def save_wrong_images(args, results_raw, log_writer, data_loader_test, max_saved = 30):

    dataset_test = data_loader_test.dataset

    # Get predictions and ground truth
    image_predictions = results_raw['predictions']
    image_gt = results_raw['true_labels']
    image_predictions_label = np.array(image_predictions).argmax(axis=1)
    image_gt_label = np.array(image_gt).argmax(axis=1)

    # Get wrong predictions indices
    wrong_predictions = np.where(image_predictions_label != image_gt_label)[0]
    wrong_predictions = wrong_predictions.tolist()
    wrong_indices = [int(idx) for idx in wrong_predictions]
    random.shuffle(wrong_indices)

    # Image IDs
    # image_ids = results_raw['img_names']
    img_path_rel = results_raw['img_path_rel']
    img_path_abs = results_raw['img_path_abs']

    # Make plots of wrong images
    if max_saved > len(wrong_indices):
        max_saved = len(wrong_indices)
    for i in wrong_indices[:max_saved]:

        fig = make_plot_of_wrong_image(dataset_test[i][0], img_path_abs[i], image_predictions_label[i], image_gt_label[i], image_predictions[i], args)

        # Draw figure on canvas
        fig.canvas.draw()

        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:,:,0:2]

        # Save figure to tensorboard
        log_writer.add_figure(f'wrong_images/{img_path_rel[i]}', fig, 0)

def visualize_augmentations(args, data_loader_train, log_writer, max_saved = 30):

    dataset_train = data_loader_train.dataset

    # Make plots of augemntations
    if max_saved > len(dataset_train):
        max_saved = len(dataset_train)
    for i in range(max_saved):

        fig = make_plot_of_augmentations(dataset_train[i], args)

        # Draw figure on canvas
        fig.canvas.draw()

        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:,:,0:2]

        # Save figure to tensorboard
        log_writer.add_figure(f'train_images/{i}', fig, 0)

    return

def make_confusion_matrix(args, results_raw, log_writer, classes):

    # Get predictions and ground truth
    image_predictions = results_raw['predictions']
    image_gt = results_raw['true_labels']
    image_predictions_label = np.array(image_predictions).argmax(axis=1)
    image_gt_label = np.array(image_gt).argmax(axis=1)

    #  Make confusion matrix
    fig = plot_confusion_matrix(list(image_gt_label), list(image_predictions_label), labels=classes, title='Confusion matrix', tensor_name = 'test/confusion_matrix', normalize=False)

    # Save figure to tensorboard
    log_writer.add_figure(f'confusion_matrix', fig, 0)

    return

def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    ''' 
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
        normalize = False               : If False, plot the raw numbers, If True, plot the proportions in rounded percentage.

    Returns:
        summary: TensorFlow summary 

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    # cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype('int')
        cm = cm.astype('int')
        cm_norm = cm
    else:
        cm_norm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype('int')
        cm_norm = cm_norm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm_norm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=30)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=10, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=30)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=10, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    # summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return fig

def create_average_metrics(args, global_rank):
    # Create dataframe with average metrics
    val_data_avg, test_data_avg, hparams = make_mean(args.task)

    # Create tensorboard writer
    args.task = os.path.join(args.task, 'average')
    writer = init_log_writer(args = args, global_rank = dist.get_rank())

    # Save dataframe to tensorboard
    save_dataframe_to_tensorboard(writer, val_data_avg, test_data_avg, hparams)

    return

def save_metrics_and_images(log_writer, args, results_raw, metrics_dict_test, data_loader_train, data_loader_test):
    if is_main_process():
        # Save all metrics
        save_all_metrics(args, metrics_dict_test, log_writer, results_raw)

        # Save images
        make_confusion_matrix(args, results_raw, log_writer, data_loader_test.dataset.classes)
        if args.few_shot_learning == -1:
            visualize_augmentations(args, data_loader_train, log_writer)
            save_wrong_images(args, results_raw, log_writer, data_loader_test)
    
