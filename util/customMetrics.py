import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix, classification_report, cohen_kappa_score
from sklearn.metrics import confusion_matrix as conf_matt_sklearn

class DatasetSpecificMetrics():
    def __init__(self, args):

        self.args = args
        diabetic_retinopathy = ['aptos', 'eyepacs', 'idrid', 'messidor', 'drtid']
        other = ['papila']
        # check if any of the strings in the list is in the data_path
        if any(x in args.data_path.lower() for x in diabetic_retinopathy):
            self.dataset = 'diabetic_retinopathy'
        elif any(x in args.data_path.lower() for x in other):
            self.dataset = 'other'
        else:
            raise Exception('Dataset not supported')
        return
    
    def __call__(self, true_label_array, prediction_softmax_array, img_paths_rel, img_paths_abs, metric_logger, data_loader, n_samples_per_class, mode):
        if self.dataset == 'diabetic_retinopathy':

            # Get class names
            classes = data_loader.dataset.classes

            # Detach n_samples_per_class
            n_samples_per_class = n_samples_per_class.detach().cpu().numpy().tolist()

            # Get raw results
            results_raw = self.get_raw_results(true_label_array, prediction_softmax_array, img_paths_rel, img_paths_abs)

            # Get Training metrics
            self.training_metrics(mode, n_samples_per_class)

            # Get basic multi-class metrics
            metrics_dict = self.multiclass(true_label_array, prediction_softmax_array, metric_logger, classes, n_samples_per_class)

            return metrics_dict, results_raw
        
        elif self.dataset == 'other':

            # Get class names
            classes = data_loader.dataset.classes

            # Detach n_samples_per_class
            n_samples_per_class = n_samples_per_class.detach().cpu().numpy().tolist()

            # Get raw results
            results_raw = self.get_raw_results(true_label_array, prediction_softmax_array, img_paths_rel, img_paths_abs)

            # Get Training metrics
            self.training_metrics(mode, n_samples_per_class)

            # Get basic glaucoma metrics
            metrics_dict = self.glaucoma(true_label_array, prediction_softmax_array, metric_logger, classes, n_samples_per_class)

            return metrics_dict, results_raw

        else:
            raise Exception('Dataset not supported')
        
    def multiclass(self, true_label_array, prediction_softmax_array, metric_logger, classes, n_samples_per_class, prefix=""):

        num_class = len(classes)

        true_label_array_index = np.argmax(true_label_array, axis=1)
        prediction_softmax_array_index = np.argmax(prediction_softmax_array, axis=1)

        confusion_matrix = conf_matt_sklearn(true_label_array_index, prediction_softmax_array_index,labels=[i for i in range(num_class)])
        classification_report_ = classification_report(true_label_array_index, prediction_softmax_array_index,labels=[i for i in range(num_class)], zero_division=0.0, output_dict=False, target_names=classes)
        classification_report_dict = classification_report(true_label_array_index, prediction_softmax_array_index,labels=[i for i in range(num_class)], zero_division=0.0, output_dict=True, target_names=classes)
        acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
        f1_macro = f1_score(true_label_array_index, prediction_softmax_array_index, average='macro')
        f1_weighted = f1_score(true_label_array_index, prediction_softmax_array_index, average='weighted')

        auc_roc = roc_auc_score(true_label_array, prediction_softmax_array,multi_class='ovr',average='macro')
        auc_pr = average_precision_score(true_label_array, prediction_softmax_array,average='macro')
        kappa = cohen_kappa_score(true_label_array_index, prediction_softmax_array_index)
        lkappa = cohen_kappa_score(true_label_array_index, prediction_softmax_array_index,weights='linear')
        qkappa = cohen_kappa_score(true_label_array_index, prediction_softmax_array_index,weights='quadratic')

        ##########################################
        ## rDR metrics
        # Combine Class 0 and 1 and Class 2, 3, 4 for true labels
        true_label_array_rDR = np.zeros((true_label_array.shape[0], 2))
        true_label_array_rDR[:,0] = true_label_array[:,0] + true_label_array[:,1]
        true_label_array_rDR[:,1] = true_label_array[:,2] + true_label_array[:,3] + true_label_array[:,4]

        # Combine Class 0 and 1 and Class 2, 3, 4 for predictions
        prediction_softmax_array_rDR = np.zeros((prediction_softmax_array.shape[0], 2))
        prediction_softmax_array_rDR[:,0] = prediction_softmax_array[:,0] + prediction_softmax_array[:,1]
        prediction_softmax_array_rDR[:,1] = prediction_softmax_array[:,2] + prediction_softmax_array[:,3] + prediction_softmax_array[:,4]

        # Get new n_samples_per_class
        n_samples_per_class_rDR = np.zeros((2))
        n_samples_per_class_rDR[0] = n_samples_per_class[0] + n_samples_per_class[1]
        n_samples_per_class_rDR[1] = n_samples_per_class[2] + n_samples_per_class[3] + n_samples_per_class[4]

        # Classes rDR
        classes_rDR = ['No_Referral', 'Referral']

        # Get new confusion matrix
        true_label_array_rDR_index = np.argmax(true_label_array_rDR, axis=1)
        prediction_softmax_array_rDR_index = np.argmax(prediction_softmax_array_rDR, axis=1)

        confusion_matrix_rDR = conf_matt_sklearn(true_label_array_rDR_index, prediction_softmax_array_rDR_index)
        classification_report_rDR = classification_report(true_label_array_rDR_index, prediction_softmax_array_rDR_index,labels=[i for i in range(len(classes_rDR))], zero_division=0.0, output_dict=False, target_names=classes_rDR)
        classification_report_rDR_dict = classification_report(true_label_array_rDR_index, prediction_softmax_array_rDR_index,labels=[i for i in range(len(classes_rDR))], zero_division=0.0, output_dict=True, target_names=classes_rDR)
        acc_rDR, sensitivity_rDR, specificity_rDR, precision_rDR, G_rDR, F1_rDR, mcc_rDR = misc_measures(confusion_matrix_rDR)
        f1_macro_rDR = f1_score(true_label_array_rDR_index, prediction_softmax_array_rDR_index, average='macro')
        f1_weighted_rDR = f1_score(true_label_array_rDR_index, prediction_softmax_array_rDR_index, average='weighted')

        auc_roc_rDR = roc_auc_score(true_label_array_rDR, prediction_softmax_array_rDR,multi_class='ovr',average='macro')
        auc_pr_rDR = average_precision_score(true_label_array_rDR, prediction_softmax_array_rDR,average='macro')

        print('\n' + str(confusion_matrix_rDR))
        print('\n' + str(classification_report_rDR))
        print('\n' + str(confusion_matrix))
        print('\n' + str(classification_report_))

        ##########################################

        ## Print metrics
        print('{} Average Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} Kappa: {:.4f} QKappa: {:.4f} mean F1-score: {:.4f} MCC: {:.4f} rDR_Acc {:.4f} rDR_AUC_ROC {:.4f}'.format(prefix, acc, auc_roc, auc_pr, kappa, qkappa, np.mean(F1), np.mean(mcc), acc_rDR, auc_roc_rDR))

        # Gather metrics to save
        metrics_dict = {'acc': float(metric_logger.meters['acc1'].global_avg/100),
                        'rDR_acc': float(acc_rDR),
                        'acc2': float(metric_logger.meters['acc2'].global_avg/100),
                        'loss': float(metric_logger.meters['loss'].global_avg),
                        'sensitivity': sensitivity.tolist(),
                        'specificity': specificity.tolist(),
                        'precision': precision.tolist(),
                        'auc_roc': float(auc_roc),
                        'auc_pr': float(auc_pr),
                        'rDR_auc_roc': float(auc_roc_rDR),
                        'rDR_auc_pr': float(auc_pr_rDR),
                        'F1_mean': float(f1_macro),
                        'F1_weighted': float(f1_weighted),
                        'rDR_F1_mean': float(f1_macro_rDR),
                        'rDR_F1_weighted': float(f1_weighted_rDR),
                        'F1': F1.tolist(),
                        'mcc': mcc.tolist(),
                        'MCC_mean': float(np.mean(mcc)),
                        'rDR_MCC_mean': float(np.mean(mcc_rDR)),
                        'kappa': float(kappa),
                        'lkappa': float(lkappa),
                        'qkappa': float(qkappa),
                        'confusion_matrix': confusion_matrix.tolist(),
                        'classification_report': classification_report_dict,
                        'n_samples_per_class': n_samples_per_class,
                        }
        
        return metrics_dict
    
    def glaucoma(self, true_label_array, prediction_softmax_array, metric_logger, classes, n_samples_per_class, prefix=""):

        num_class = len(classes)

        true_label_array_index = np.argmax(true_label_array, axis=1)
        prediction_softmax_array_index = np.argmax(prediction_softmax_array, axis=1)

        confusion_matrix = conf_matt_sklearn(true_label_array_index, prediction_softmax_array_index,labels=[i for i in range(num_class)])
        classification_report_ = classification_report(true_label_array_index, prediction_softmax_array_index,labels=[i for i in range(num_class)], zero_division=0.0, output_dict=False, target_names=classes)
        classification_report_dict = classification_report(true_label_array_index, prediction_softmax_array_index,labels=[i for i in range(num_class)], zero_division=0.0, output_dict=True, target_names=classes)
        acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
        f1_macro = f1_score(true_label_array_index, prediction_softmax_array_index, average='macro')
        f1_weighted = f1_score(true_label_array_index, prediction_softmax_array_index, average='weighted')

        auc_roc = roc_auc_score(true_label_array, prediction_softmax_array,multi_class='ovr',average='macro')
        auc_pr = average_precision_score(true_label_array, prediction_softmax_array,average='macro')
        kappa = cohen_kappa_score(true_label_array_index, prediction_softmax_array_index)
        lkappa = cohen_kappa_score(true_label_array_index, prediction_softmax_array_index,weights='linear')
        qkappa = cohen_kappa_score(true_label_array_index, prediction_softmax_array_index,weights='quadratic')

        print('\n' + str(confusion_matrix))
        print('\n' + str(classification_report_))

        ## Print metrics
        print('{} Average Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} Kappa: {:.4f} QKappa: {:.4f} mean F1-score: {:.4f} MCC: {:.4f}'.format(prefix, acc, auc_roc, auc_pr, kappa, qkappa, np.mean(F1), np.mean(mcc)))

        # Gather metrics to save
        metrics_dict = {'acc': float(metric_logger.meters['acc1'].global_avg/100),
                        'acc2': float(metric_logger.meters['acc2'].global_avg/100),
                        'loss': float(metric_logger.meters['loss'].global_avg),
                        'sensitivity': sensitivity.tolist(),
                        'specificity': specificity.tolist(),
                        'precision': precision.tolist(),
                        'auc_roc': float(auc_roc),
                        'auc_pr': float(auc_pr),
                        'F1_mean': float(f1_macro),
                        'F1_weighted': float(f1_weighted),
                        'F1': F1.tolist(),
                        'mcc': mcc.tolist(),
                        'MCC_mean': float(np.mean(mcc)),
                        'kappa': float(kappa),
                        'lkappa': float(lkappa),
                        'qkappa': float(qkappa),
                        'confusion_matrix': confusion_matrix.tolist(),
                        'classification_report': classification_report_dict,
                        'n_samples_per_class': n_samples_per_class,
                        }
        
        return metrics_dict
    
    def training_metrics(self, mode, n_samples_per_class, prefix = 'multiclass'):

        print("{} Data Composition: ".format(mode) + str(n_samples_per_class))
        
        return
    
    def get_raw_results(self, true_label_array, prediction_softmax_array, img_paths_rel, img_paths_abs):

        results_raw = {'predictions': np.array(prediction_softmax_array).tolist(),
                    'true_labels': np.array(true_label_array).tolist(),
                    'img_path_rel': img_paths_rel,
                    'img_path_abs': img_paths_abs}
        
        return results_raw


def misc_measures(confusion_matrix):
    
    num_classes = confusion_matrix.shape[0]
    
    acc = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    G = np.zeros(num_classes)
    F1_score_2 = np.zeros(num_classes)
    mcc_ = np.zeros(num_classes)
    
    for i in range(num_classes):
        true_positive = confusion_matrix[i, i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        true_negative = np.sum(confusion_matrix) - true_positive - false_positive - false_negative

        sensitivity[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity[i] = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        G[i] = np.sqrt(sensitivity[i] * specificity[i])
        F1_score_2[i] = 2 * precision[i] * sensitivity[i] / (precision[i] + sensitivity[i]) if (precision[i] + sensitivity[i]) > 0 else 0

        mcc_[i] = (true_positive * true_negative - false_positive * false_negative) / np.sqrt(
            (true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)
        ) if (true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative) > 0 else 0

    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
    