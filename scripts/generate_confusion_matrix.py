import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
from textwrap import wrap
import itertools

from sklearn.metrics import confusion_matrix
import sys


def make_confusion_matrix(results_raw, classes, title='Confusion matrix', file_name='confusion_matrix.png', base_folder = 'csv_results/'):

    # Get predictions and ground truth
    image_predictions = results_raw['predictions']
    image_gt = results_raw['true_labels']
    # convert string to list
    image_predictions = image_predictions.str.replace('[', '').str.replace(']', '').str.replace(' ', '').str.replace('\n', '')
    image_gt = image_gt.str.replace('[', '').str.replace(']', '').str.replace(' ', '').str.replace('\n', '')

    # get column value of max value in each row
    image_predictions = image_predictions.str.split(',').apply(lambda x: [float(i) for i in x])
    image_gt = image_gt.str.split(',').apply(lambda x: [float(i) for i in x])

    #get index of max value in each row
    image_predictions_label = image_predictions.apply(lambda x: x.index(max(x)))
    image_gt_label = image_gt.apply(lambda x: x.index(max(x)))

    #  Make confusion matrix
    fig = plot_confusion_matrix(list(image_gt_label), list(image_predictions_label), labels=classes, title = title, tensor_name = 'test/confusion_matrix', normalize=False)

    file_name = file_name + '.png'
    fig.savefig(os.path.join(base_folder, file_name))

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

    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype('int')
        cm = cm.astype('int')
        cm_norm = cm
    else:
        cm_norm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype('int')
        cm_norm = cm_norm.astype('int')

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm_norm, cmap='Oranges')

    ax.set_title(title, fontsize=30)

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

    return fig


if __name__ == '__main__':
    # Load the results
    base_folder = 'csv_results/'

    #all csv files in the directory
    classes = ['no DR', 'mild DR', 'moderate DR', 'severe DR', 'proliferative DR']


    for file in os.listdir(base_folder):
        if file.endswith('.csv'):
            name = file.split('.')[0]
            # Load the results
            raw_results = pd.read_csv(os.path.join(base_folder, file))
            new_file_name = 'MSDF '+ name
            new_file_name = new_file_name.replace(' ', '_')

            make_confusion_matrix(raw_results, classes, title = name, file_name = new_file_name, base_folder = base_folder)
