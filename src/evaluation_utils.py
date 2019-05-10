import numpy as np
import itertools
from sklearn.metrics import roc_auc_score, confusion_matrix
from matplotlib import pyplot as plt


def accuracy(t_p, t_n, f_n, f_p):
    return (np.sum(t_p) + np.sum(t_n)) / (np.sum(t_p) + np.sum(t_n) + np.sum(f_n) + np.sum(f_p))


def precision(t_p, f_p):
    return np.sum(t_p) / (np.sum(t_p) + np.sum(f_p))


def recall(t_p, f_n):
    return np.sum(t_p) / (np.sum(t_p) + np.sum(f_n))


def fscore(precision, recall):
    return 2 * precision * recall / (precision + recall)


def accuracy_precision_recall_fscore(pred_probs, true_labels, num_classes):
    pred = np.array(pred_probs)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    t_p = np.logical_and(pred == 1, true_labels == 1)
    t_n = np.logical_and(pred == 0, true_labels == 0)
    f_p = np.logical_and(pred == 1, true_labels == 0)
    f_n = np.logical_and(pred == 0, true_labels == 1)

    acc_overall = accuracy(t_p, t_n, f_n, f_p)
    precision_overall = precision(t_p, f_p)
    recall_overall = recall(t_p, f_n)
    fscore_overall = fscore(precision_overall, recall_overall)
    accs = []
    precisions = []
    recalls = []
    fscores = []
    for i in range(num_classes):
        accs.append(accuracy(t_p[:, i], t_n[:, i], f_n[:, i], f_p[:, i]))
        precisions.append(precision(t_p[:, i], f_p[:, i]))
        recalls.append(recall(t_p[:, i], f_n[:, i]))
        fscores.append(fscore(precisions[i], recalls[i]))

    return acc_overall, accs, precision_overall, precisions, recall_overall, recalls, fscore_overall, fscores


def roc_auc_calculator(true_labels, pred_probs):
    labels = np.array(true_labels)
    labels[labels < 0.5] = 0
    labels = labels.astype(int)
    return roc_auc_score(labels, pred_probs, sample_weight=true_labels > -0.5)


def roc_auc(pred_probs, true_labels, num_classes):
    overall = roc_auc_calculator(true_labels.flatten(), pred_probs.flatten())
    roc_aucs = []
    for i in range(num_classes):
        roc_aucs.append(roc_auc_calculator(true_labels[:, i], pred_probs[:, i]))
    return overall, roc_aucs


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, path_fig='./{}'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    source code: "https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py"
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path_fig)


def save_confusion_matrix(pred_probs, true_labels, case_array, path_fig):
    labels = np.array(true_labels)
    labels[labels < 0.5] = 0
    labels = labels.astype(int)
    pred = np.array(pred_probs)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    for i in range(len(case_array)):
        cm = confusion_matrix(labels[:, i], pred[:, i], sample_weight=true_labels[:, i] > -0.5)
        plot_confusion_matrix(cm, ['0', '1'], normalize=True, title=case_array[i], path_fig=path_fig.format(case_array[i]))


def plot_loss_curve(loss_arrays, labels, title, path_fig, x_label):
    fig = plt.figure()
    fig.set_figwidth(fig.get_figwidth() * 3)
    fig.set_figheight(fig.get_figheight())
    for i in range(len(loss_arrays)):
        plt.plot(loss_arrays[i], label=labels[i])

    plt.ylabel('$\mathcal{L}(W)$', fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.title(title)
    plt.legend()
    plt.savefig(path_fig)
